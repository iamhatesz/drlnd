import os
import random
from collections import namedtuple
from copy import copy
from time import sleep
from typing import Optional, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

from utils.logging import init_tensorboard_logger, get_run_id
from utils.metrics import Metrics

State = torch.tensor
TerminalState = None
Action = torch.tensor
Reward = float

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
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


class OrnsteinUhlenbackProcess:
    def __init__(self, dims: int, mu_start: float = 1.0, mu_final: float = 0.0,
                 theta: float = 0.02, sigma: float = 1.0,
                 dt: float = 0.01):
        self.dims = dims
        self.mu_start = mu_start * np.ones(dims)
        self.mu_final = mu_final * np.ones(dims)
        self.theta = theta
        self.sigma = sigma
        self.time = 0.0
        self.dt = dt
        self.xt = copy(self.mu_start)

    def reset(self):
        self.xt = copy(self.mu_start)

    def sample(self):
        self.time += self.dt
        dx = self.theta * (self.mu_final - self.xt) * self.dt + self.sigma * np.random.normal(size=self.dims) * self.dt
        self.xt += dx
        return self.xt


def hidden_unit(layer):
    inp = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(inp)
    return -lim, lim


class ActorNetwork(nn.Module):
    def __init__(self, state_space: int, action_space: int):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, action_space)
        self.init_weights()

    def init_weights(self):
        self.fc1.weight.data.uniform_(*hidden_unit(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return x


class CriticNetwork(nn.Module):
    def __init__(self, state_space: int, action_space: int):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256 + action_space, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.init_weights()

    def init_weights(self):
        self.fc1.weight.data.uniform_(*hidden_unit(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_unit(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_unit(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        x = F.leaky_relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DDPG:
    def __init__(self, env: UnityEnvironment, state_space: int, action_space: int,
                 actor_network_builder: Callable[[int, int], nn.Module],
                 critic_network_builder: Callable[[int, int], nn.Module],
                 gamma: float, batch_size: int, target_update: int, use_soft_updates: bool,
                 policy_update: int, num_policy_updates: int,
                 min_samples_in_memory: int, memory_capacity: int,
                 device: str, tb_logger: SummaryWriter):
        self._device = torch.device(device)
        self._dtype = torch.float

        self._actor = actor_network_builder(state_space, action_space).to(device=self._device, dtype=self._dtype).train()
        self._actor_target = actor_network_builder(state_space, action_space).to(device=self._device, dtype=self._dtype)
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._actor_target.eval()

        self._critic = critic_network_builder(state_space, action_space).to(device=self._device, dtype=self._dtype).train()
        self._critic_target = critic_network_builder(state_space, action_space).to(device=self._device, dtype=self._dtype)
        self._critic_target.load_state_dict(self._critic.state_dict())
        self._critic_target.eval()

        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=1e-4)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=1e-4, weight_decay=0)

        self._env = env
        self._state_space = state_space
        self._action_space = action_space
        self._brain_name = self._env.brain_names[0]

        self._memory = ReplayMemory(memory_capacity)
        self._metrics = Metrics()

        self._gamma = gamma
        self._batch_size = batch_size
        self._target_update = target_update
        self._use_soft_update = use_soft_updates
        self._policy_update = policy_update
        self._num_policy_updates = num_policy_updates
        self._min_samples_in_memory = min_samples_in_memory

        self._noise = OrnsteinUhlenbackProcess(dims=self._action_space, mu_start=1., mu_final=0., theta=0.015, sigma=0.2, dt=0.005)

        self._tb = tb_logger

    def control(self, state: State, explore: bool = False) -> Action:
        self._actor.eval()
        with torch.no_grad():
            action = self._actor(state)

        self._actor.train()
        if explore and self._metrics.step < 1000000:
            noise = self._noise.sample()
            action += torch.tensor(noise, device=self._device, dtype=self._dtype)
            for i, action_noise in enumerate(noise.tolist()):
                self._tb.add_scalar(f'noise/a{i}', float(action_noise), self._metrics.step)

        return torch.clamp(action, -1, 1)

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
                action = self.control(state, explore=True)
                env_info = self._env.step(action.cpu().numpy())[self._brain_name]
                next_state = torch.tensor(env_info.vector_observations[0], device=self._device, dtype=self._dtype)
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                self._metrics.next_step(reward)

                if done:
                    next_state = TerminalState

                self._store_transition(state, action, next_state, reward)

                if self._metrics.last_step() % self._policy_update == 0:
                    for _ in range(self._num_policy_updates):
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
        assert action_batch.size() == torch.Size([self._batch_size, self._action_space])
        reward_batch = torch.stack(batch.reward).to(self._device)
        assert reward_batch.size() == torch.Size([self._batch_size, 1])

        state_action_values = self._critic(state_batch, action_batch)
        assert state_action_values.size() == torch.Size([self._batch_size, 1])

        next_state_values = torch.zeros(self._batch_size, device=self._device, dtype=self._dtype)
        # Use `network` to choose an action and `target_network` to compute its value
        next_state_values[non_final_mask] = self._critic_target(non_final_next_states,
                                                                self._actor_target(non_final_next_states)).squeeze()

        expected_state_action_values = (next_state_values.unsqueeze(1) * self._gamma) + reward_batch
        assert expected_state_action_values.size() == torch.Size([self._batch_size, 1])

        critic_loss = F.mse_loss(state_action_values, expected_state_action_values)
        self._tb.add_scalar('ddpg/critic_loss', float(critic_loss), self._metrics.step)

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1.)
        self._critic_optimizer.step()

        actor_loss = -self._critic(state_batch, self._actor(state_batch)).mean()
        self._tb.add_scalar('ddpg/actor_loss', float(actor_loss), self._metrics.step)

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 10.)
        self._actor_optimizer.step()

        self._update_weights(self._actor, self._actor_target)
        self._update_weights(self._critic, self._critic_target)

    def _store_transition(self, state: State, action: Action, next_state: State, reward: Reward):
        self._memory.push(state.to(device=self._device),
                          action.to(device=self._device),
                          next_state.to(device=self._device) if next_state is not TerminalState else next_state,
                          torch.tensor([reward], device=self._device, dtype=self._dtype))
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
        torch.save(self._actor.state_dict(),
                   os.path.join(checkpoint_dir, f'episode{episode:04d}_score{score:04.2f}.weights'))

    def _load_checkpoint(self, checkpoint: str):
        self._actor.load_state_dict(torch.load(checkpoint))
        self._actor.eval()


if __name__ == '__main__':
    SEED = 1337
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = UnityEnvironment(file_name=os.path.join(os.curdir, 'bin', 'Reacher.x86_64'), no_graphics=False)
    tb = init_tensorboard_logger(os.path.join(os.pardir, 'tensorboard'), get_run_id())
    algo = DDPG(env=env, state_space=33, action_space=4,
                actor_network_builder=ActorNetwork,
                critic_network_builder=CriticNetwork,
                gamma=0.99, batch_size=128, target_update=100, use_soft_updates=True,
                policy_update=2, num_policy_updates=2,
                min_samples_in_memory=2**13, memory_capacity=2**20,
                device='cuda', tb_logger=tb)

    # Uncomment to train
    # algo.train()

    # Uncomment to evaluate
    algo.eval(r'./episode1258_score39.14.weights',
              num_episodes=100,
              step_delay=None)
    env.close()
