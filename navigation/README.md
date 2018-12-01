# Udacity Deep Reinforcement Learning Nanodegree
## Project 1: Navigation

### Project details
The goal of this project was to train RL agent to navigate in a large, square world and collect bananas. 
For each yellow banana collected the agent receives +1 reward, while for each blue banana the agent receives -1.
Thus, the agent has to learn how to collect as many yellow bananas as possible and avoiding contact with blue bananas.

The state space consists of 37 dimensions, which describes agent's velocity and ray-based perception of objects 
around the agent's forward direction.

The action space consists of 4 discrete actions: move forward, move backward, turn left, and turn right.

The task is episodic and it is considered solved when the agent reaches average score of at least +13 over 100
consecutive episodes.

### Getting started
In order to setup this project, you have to download Banana environment from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector)
and place it into `bin/` directory. If you are using Windows or Mac OS X, 
then you should also adjust the path in `dqn.py`.

To install Python dependencies you can simply run `pip install -r requirements.txt` from the parent directory.

### Instructions
To train the agent uncomment following lines in `dqn.py`:

``
algo.train()
``

To evaluate the agent uncomment following lines in `dqn.py` and specify checkpoint:

``
algo.eval(r'./episode0627_score15.58.weights',
              num_episodes=100,
              step_delay=None)
``

You can specify number of episodes to run with `num_episodes`, 
and whether to delay execution of each step with `step_delay`.

### Report
You can read detailed report [here](https://github.com/iamhatesz/drlnd/blob/master/navigation/Report.md).
