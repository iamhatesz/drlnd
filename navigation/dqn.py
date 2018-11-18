import os
import random
from collections import namedtuple
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

from utils.logging import init_tensorboard_logger, get_run_id
from utils.math import exp_decay
from utils.metrics import Metrics

State = torch.tensor
TerminalState = None
Action = torch.tensor
Reward = float

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._memory = []
        self._index = 0

    def __len__(self):
        return len(self._memory)

    def push(self, state: State, action: Action, next_state: State, reward: Reward):
        t = Transition(state, action, next_state, reward)
        if len(self._memory) < self.capacity:
            self._memory.append(t)
        else:
            self._memory[self._index] = t
        self._index = (self._index + 1) % self.capacity

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

        self._env = env
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
                return self._network(state).max(dim=0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], device=self._device, dtype=torch.long)

    def train(self):
        while True:
            env_info = self._env.reset(train_mode=True)[self._brain_name]
            state = torch.tensor(env_info.vector_observations[0], device=self._device, dtype=self._dtype)

            while True:
                action = self.control(state)
                env_info = self._env.step(action.numpy())[self._brain_name]
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

    def _store_transition(self, state: State, action: Action, next_state: State, reward: Reward):
        self._memory.push(state, action, next_state, reward)
        self._tb.add_scalar('dqn/replay_memory_size', len(self._memory), self._metrics.step)

    @classmethod
    def _update_weights(cls, src: nn.Module, dst: nn.Module):
        dst.load_state_dict(src.state_dict())


if __name__ == '__main__':
    env = UnityEnvironment(file_name=os.path.join(os.curdir, 'bin', 'Banana.x86_64'))
    tb = init_tensorboard_logger(os.path.join(os.pardir, 'tensorboard'), get_run_id())
    algo = DQN(env=env, state_space=37, action_space=4,
               network_builder=DeepQNetwork,
               gamma=0.99, batch_size=64, target_update=10, memory_capacity=10000,
               eps_fn=exp_decay(0.9, 0.05, 1000),
               device='cpu', tb_logger=tb)

    algo.train()
    env.close()
