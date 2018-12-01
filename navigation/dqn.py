import os
import random
from collections import namedtuple
from time import sleep
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

from utils.logging import init_tensorboard_logger, get_run_id
from utils.math import linear_decay
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

    def __len__(self) -> int:
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
                 use_double_dqn: bool,
                 gamma: float, batch_size: int, target_update: int, use_soft_updates: bool, policy_update: int,
                 min_samples_in_memory: int, memory_capacity: int, eps_fn: Callable[[int], float],
                 device: str, tb_logger: SummaryWriter):
        self._device = torch.device(device)
        self._dtype = torch.float
        self._network = network_builder(state_space, action_space).type(self._dtype).to(self._device).train()
        self._target_network = network_builder(state_space, action_space).type(self._dtype).to(self._device)
        self._target_network.load_state_dict(self._network.state_dict())
        self._target_network.eval()

        self._optimizer = optim.Adam(self._network.parameters())

        self._env = env
        self._state_space = state_space
        self._action_space = action_space
        self._brain_name = self._env.brain_names[0]

        self._memory = ReplayMemory(memory_capacity)
        self._metrics = Metrics()

        self._use_double_dqn = use_double_dqn

        self._gamma = gamma
        self._batch_size = batch_size
        self._target_update = target_update
        self._use_soft_update = use_soft_updates
        self._policy_update = policy_update
        self._min_samples_in_memory = min_samples_in_memory

        self._eps = eps_fn
        self._tb = tb_logger

    def control(self, state: State, explore: bool = True) -> Action:
        eps_threshold = self._eps(self._metrics.step)
        self._tb.add_scalar('dqn/epsilon', float(eps_threshold), self._metrics.step)
        if random.random() > eps_threshold or not explore:
            with torch.no_grad():
                return self._network(state).max(dim=0)[1].view(1)
        else:
            return torch.tensor([random.randrange(self._action_space)], device='cpu', dtype=torch.long)

    def eval(self, checkpoint: str, num_episodes: int = 100, step_delay: Optional[float] = None):
        self._load_checkpoint(checkpoint)

        while True:
            env_info = self._env.reset(train_mode=True)[self._brain_name]
            state = torch.tensor(env_info.vector_observations[0], device=self._device, dtype=self._dtype)

            while True:
                action = self.control(state, explore=False)
                env_info = self._env.step(action.cpu().numpy())[self._brain_name]
                next_state = torch.tensor(env_info.vector_observations[0], device=self._device, dtype=self._dtype)
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                self._metrics.next_step(reward)

                state = next_state

                if step_delay:
                    sleep(step_delay)

                if done:
                    self._metrics.next_episode()
                    print(f'Mean reward after {self._metrics.episode} episodes '
                          f'= {self._metrics.mean_episode_reward(horizon=100)}')
                    break

            if self._metrics.episode >= num_episodes:
                break

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

                self._metrics.next_step(reward)

                if done:
                    next_state = TerminalState

                self._store_transition(state, action, next_state, reward)

                if self._metrics.last_step() % self._policy_update == 0:
                    self._learn()

                state = next_state

                if done:
                    self._metrics.next_episode()
                    self._tb.add_scalar('stat/episode_reward',
                                        self._metrics.last_episode_reward(),
                                        self._metrics.last_episode())
                    self._tb.add_scalar('stat/mean_episode_reward10',
                                        self._metrics.mean_episode_reward(horizon=10),
                                        self._metrics.last_episode())
                    benchmark100 = self._metrics.mean_episode_reward(horizon=100)
                    self._tb.add_scalar('stat/mean_episode_reward100',
                                        benchmark100,
                                        self._metrics.last_episode())

                    if benchmark100 > self._metrics.baseline:
                        self._metrics.new_baseline(benchmark100)
                        self._save_checkpoint(self._metrics.episode, benchmark100)
                    break

    def _learn(self):
        if len(self._memory) < self._min_samples_in_memory:
            return

        transitions = self._memory.sample(self._batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not TerminalState, batch.next_state)),
                                      device=self._device, dtype=torch.uint8)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not TerminalState]).to(self._device)
        assert 0 <= non_final_next_states.size(0) <= self._batch_size
        assert non_final_next_states.size(1) == self._state_space

        state_batch = torch.stack(batch.state).to(self._device)
        assert state_batch.size() == torch.Size([self._batch_size, self._state_space])
        action_batch = torch.stack(batch.action).to(self._device)
        assert action_batch.size() == torch.Size([self._batch_size, 1])
        reward_batch = torch.stack(batch.reward).to(self._device)
        assert reward_batch.size() == torch.Size([self._batch_size, 1])

        state_action_values = self._network(state_batch).gather(1, action_batch)
        assert state_action_values.size() == torch.Size([self._batch_size, 1])

        next_state_values = torch.zeros(self._batch_size, device=self._device, dtype=self._dtype)

        if self._use_double_dqn:
            # Use `network` to choose an action and `target_network` to compute its value
            next_state_values[non_final_mask] = self._target_network(non_final_next_states).detach().gather(
                1, self._network(non_final_next_states).detach().max(dim=1, keepdim=True)[1]
            ).squeeze()
        else:
            # Use greedy action based on its value estimated using `target_network`
            next_state_values[non_final_mask] = self._target_network(non_final_next_states).max(
                dim=1, keepdims=True
            )[0].detach()

        expected_state_action_values = (next_state_values.unsqueeze(1) * self._gamma) + reward_batch
        assert expected_state_action_values.size() == torch.Size([self._batch_size, 1])

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self._tb.add_scalar('dqn/loss', float(loss), self._metrics.step)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), 10.)
        self._optimizer.step()

        self._update_weights(self._network, self._target_network)

    def _store_transition(self, state: State, action: Action, next_state: State, reward: Reward):
        self._memory.push(state.to('cpu'),
                          action.to('cpu'),
                          next_state.to('cpu') if next_state is not TerminalState else next_state,
                          torch.tensor([reward], device='cpu', dtype=self._dtype))
        self._tb.add_scalar('dqn/replay_memory_size', len(self._memory), self._metrics.step)
        self._tb.add_scalar('dqn/replay_memory_index', self._memory.index, self._metrics.step)

    def _update_weights(self, src: nn.Module, dst: nn.Module):
        if self._use_soft_update:
            tau = 1. / self._target_update
            for sp, dp in zip(src.parameters(), dst.parameters()):
                dp.data.copy_(tau * sp.data + (1.0 - tau) * dp.data)
        else:
            dst.load_state_dict(src.state_dict())

    def _save_checkpoint(self, episode: int, score: float):
        checkpoint_dir = os.path.join(os.curdir, 'output', get_run_id())
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self._network.state_dict(),
                   os.path.join(checkpoint_dir, f'episode{episode:04d}_score{score:04.2f}.weights'))

    def _load_checkpoint(self, checkpoint: str):
        self._network.load_state_dict(torch.load(checkpoint))
        self._network.eval()


if __name__ == '__main__':
    env = UnityEnvironment(file_name=os.path.join(os.curdir, 'bin', 'Banana.x86_64'))
    tb = init_tensorboard_logger(os.path.join(os.pardir, 'tensorboard'), get_run_id())
    algo = DQN(env=env, state_space=37, action_space=4,
               network_builder=DeepQNetwork,
               use_double_dqn=True,
               gamma=0.99, batch_size=64, target_update=1000, use_soft_updates=True, policy_update=4,
               min_samples_in_memory=64, memory_capacity=100000, eps_fn=linear_decay(1.0, 0.05, 100000),
               device='cuda', tb_logger=tb)

    # Uncomment to train
    # algo.train()

    # Uncomment to evaluate
    algo.eval(r'./episode0627_score15.58.weights',
              num_episodes=100,
              step_delay=None)
    env.close()
