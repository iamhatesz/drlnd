from statistics import mean


class Metric(object):
    def __init__(self):
        self.step = 0
        self.step_in_episode = 0
        self.episode = 0
        self.episode_rewards = []
        self.episodes_rewards = []

    def next_step(self, reward: float):
        self.episode_rewards.append(reward)
        self.step += 1
        self.step_in_episode += 1
        return self

    def next_episode(self):
        self.step_in_episode = 0
        self.episode += 1
        self.episodes_rewards.append(self.episode_reward())
        self.episode_rewards = []

    def episode_reward(self) -> float:
        return sum(self.episode_rewards)

    def mean_episode_reward(self, horizon: int) -> float:
        return mean(self.episodes_rewards[-horizon:])