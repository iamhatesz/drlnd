import os
import random
from collections import namedtuple
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from unityagents import UnityEnvironment

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
            self._index += 1
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
    def __init__(self, network_builder: Callable[[int, int], nn.Module], state_space: int, action_space: int,
                 gamma: float, batch_size: int, target_update: int, memory_capacity: int,
                 device: str = 'cpu'):
        self._device = torch.device(device)
        self._dtype = torch.float
        self._network = network_builder(state_space, action_space).type(self._dtype).to(self._device)
        self._target_network = network_builder(state_space, action_space).type(self._dtype).to(self._device)
        self._update_weights(self._network, self._target_network)

        self._memory = ReplayMemory(memory_capacity)

        self._gamma = gamma
        self._batch_size = batch_size
        self._target_update = target_update

    def control(self, state: State) -> Action:
        pass

    def train(self):
        pass

    def learn(self):
        if len(self._memory) < self._batch_size:
            return

    def store_transition(self, state: State, action: Action, next_state: State, reward: Reward):
        self._memory.push(state, action, next_state, reward)

    @classmethod
    def _update_weights(cls, src: nn.Module, dst: nn.Module):
        dst.load_state_dict(src.state_dict())


if __name__ == '__main__':
    algo = DQN(network_builder=DeepQNetwork,
               state_space=37, action_space=4,
               gamma=0.99, batch_size=64, target_update=10, memory_capacity=10000,
               device='cpu')

    env = UnityEnvironment(file_name=os.path.join(os.curdir, 'bin', 'Banana.app'))
    brain_name = env.brain_names[0]

    env_info = env.reset(train_mode=True)[brain_name]
    state = torch.tensor(env_info.vector_observations[0])

    while True:
        action = algo.control(state)
        env_info = env.step(action)[brain_name]
        next_state = torch.tensor(env_info.vector_observations[0])
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        if done:
            next_state = TerminalState

        algo.store_transition(state, action, next_state, reward)
        algo.learn()

        state = next_state

        if done:
            break

    env.close()
