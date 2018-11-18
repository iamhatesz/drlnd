import os
import random
from collections import namedtuple
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

from utils.logging import init_tensorboard_logger, get_run_id
from utils.math import exp_decay, linear_decay
from utils.metrics import Metrics

State = torch.tensor
TerminalState = None
Action = torch.tensor
Reward = float

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.index = 0
        self._memory = []

    def __len__(self):
        return len(self._memory)

    def push(self, state: State, action: Action, next_state: State, reward: Reward):
        t = Transition(state, action, next_state, reward)
        if len(self._memory) < self.capacity:
            self._memory.append(t)
        else:
            self._memory[self.index] = t
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._memory, batch_size)


class DeepQNetwork(nn.Module):
    def __init__(self, state_space: int, action_space: int):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN(object):
    def __init__(self, env: UnityEnvironment, state_space: int, action_space: int,
                 network_builder: Callable[[int, int], nn.Module],
                 gamma: float, batch_size: int, target_update: int, memory_capacity: int,
                 eps_fn: Callable[[int], float],
                 device: str, tb_logger: SummaryWriter):
        self._device = torch.device(device)
        self._dtype = torch.float
        self._network = network_builder(state_space, action_space).type(self._dtype).to(self._device)
        self._target_network = network_builder(state_space, action_space).type(self._dtype).to(self._device)
        self._update_weights(self._network, self._target_network)

        self._optimizer = optim.Adam(self._network.parameters())

        self._env = env
        self._state_space = state_space
        self._action_space = action_space
        self._brain_name = self._env.brain_names[0]

        self._memory = ReplayMemory(memory_capacity)
        self._metrics = Metrics()

        self._gamma = gamma
        self._batch_size = batch_size
        self._target_update = target_update

        self._eps = eps_fn
        self._tb = tb_logger

    def control(self, state: State, explore: bool = True) -> Action:
        eps_threshold = self._eps(self._metrics.step)
        self._tb.add_scalar('dqn/epsilon', float(eps_threshold), self._metrics.step)
        if random.random() > eps_threshold or not explore:
            with torch.no_grad():
                return self._network(state).max(dim=0)[1].view(1)
        else:
            return torch.tensor([random.randrange(4)], device=self._device, dtype=torch.long)

    def train(self):
        while True:
            env_info = self._env.reset(train_mode=True)[self._brain_name]
            state = torch.tensor(env_info.vector_observations[0], device=self._device, dtype=self._dtype)

            while True:
                action = self.control(state)
                env_info = self._env.step(action.cpu().numpy())[self._brain_name]
                next_state = torch.tensor(env_info.vector_observations[0], device=self._device, dtype=self._dtype)
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                if done:
                    next_state = TerminalState

                self._store_transition(state, action, next_state, reward)
                self._learn()
                self._metrics.next_step(reward)

                state = next_state

                if done:
                    self._tb.add_scalar('dqn/episode_reward',
                                        self._metrics.episode_reward(),
                                        self._metrics.episode)
                    self._metrics.next_episode()
                    self._tb.add_scalar('dqn/mean_episode_reward',
                                        self._metrics.mean_episode_reward(horizon=10),
                                        self._metrics.episode)
                    break

    def _learn(self):
        if len(self._memory) < self._batch_size:
            return

        transitions = self._memory.sample(self._batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not TerminalState, batch.next_state)),
                                      device=self._device, dtype=torch.uint8)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not TerminalState])
        assert 0 <= non_final_next_states.size(0) <= self._batch_size and non_final_next_states.size(1) == self._state_space

        state_batch = torch.stack(batch.state)
        assert state_batch.size() == torch.Size([self._batch_size, self._state_space])
        action_batch = torch.stack(batch.action)
        assert action_batch.size() == torch.Size([self._batch_size, 1])
        reward_batch = torch.stack(batch.reward)
        assert reward_batch.size() == torch.Size([self._batch_size, 1])

        state_action_values = self._network(state_batch).gather(1, action_batch)
        assert state_action_values.size() == torch.Size([self._batch_size, 1])

        next_state_values = torch.zeros(self._batch_size, device=self._device, dtype=self._dtype)
        next_state_values[non_final_mask] = self._target_network(non_final_next_states).max(dim=1)[0].detach()
        expected_state_action_values = (next_state_values.unsqueeze(1) * self._gamma) + reward_batch
        assert expected_state_action_values.size() == torch.Size([self._batch_size, 1])

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self._tb.add_scalar('dqn/loss', float(loss), self._metrics.step)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), 10.)
        self._optimizer.step()

        if (self._metrics.step % self._target_update) == 0:
            self._update_weights(self._network, self._target_network)

    def _store_transition(self, state: State, action: Action, next_state: State, reward: Reward):
        self._memory.push(state, action, next_state, torch.tensor([reward], device=self._device, dtype=self._dtype))
        self._tb.add_scalar('dqn/replay_memory_size', len(self._memory), self._metrics.step)
        self._tb.add_scalar('dqn/replay_memory_index', self._memory.index, self._metrics.step)

    @classmethod
    def _update_weights(cls, src: nn.Module, dst: nn.Module):
        dst.load_state_dict(src.state_dict())


if __name__ == '__main__':
    env = UnityEnvironment(file_name=os.path.join(os.curdir, 'bin', 'Banana.x86_64'))
    tb = init_tensorboard_logger(os.path.join(os.pardir, 'tensorboard'), get_run_id())
    algo = DQN(env=env, state_space=37, action_space=4,
               network_builder=DeepQNetwork,
               gamma=0.99, batch_size=64, target_update=10, memory_capacity=50000,
               eps_fn=linear_decay(0.9, 0.05, 20000),
               device='cuda', tb_logger=tb)

    algo.train()
    env.close()
