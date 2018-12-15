# Udacity Deep Reinforcement Learning Nanodegree
## Project 2: Continuous Control

### Project details
The goal of this project was to train RL agent to control double-jointed arm so it follows given position.
For each time step that the agent's hand is in the goal location, it receives a reward of +0.1.
There are no negative rewards. 
Thus, the agent has to learn how to keep its hand into given area for as long as possible.

The state space consists of 33 dimensions, which describes position, rotation, velocity and angular velocities of the arm.

The action space consists of 4 continuous actions, which describes the torque applicable to both joints.

The task is episodic and it is considered solved when the agent reaches average score of at least +30 over 100 
consecutive episodes.

### Getting started
In order to setup this project, you have to download Reacher environment from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
and place it into `bin/` directory. If you are using Windows or Mac OS X, 
then you should also adjust the path in `ddpg.py`.

To install Python dependencies you can simply run `pip install -r requirements.txt` from the parent directory.

### Instructions
To train the agent uncomment following lines in `ddpg.py`:

``
algo.train()
``

To evaluate the agent uncomment following lines in `ddpg.py` and specify checkpoint:

``
algo.eval(r'./episode1258_score39.14.weights',
              num_episodes=100,
              step_delay=None)
``

You can specify number of episodes to run with `num_episodes`, 
and whether to delay execution of each step with `step_delay`.
