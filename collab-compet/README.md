# Udacity Deep Reinforcement Learning Nanodegree
## Project 3: Collaboration and Competition

### Project details
The goal of this project was to train a pair of RL agents to bounce a ball over a net.
If an agent hits the ball over the net, it receives a reward of +0.1.
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
Thus, the agents have to learn how to bounce a ball over the net without hitting the ground or out of bounds.

The state space for each agent consists of 24 dimensions, which contains observations of the position and velocity
of the ball and the racket.

The action space for each agent consists of 2 continuous actions, which describes the movement against the net and jumping.

The task is episodic and it is considered solved when the agent reaches average score of at least +0.5 over 100 
consecutive episodes.

### Getting started
In order to setup this project, you have to download Tennis environment from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)
and place it into `bin/` directory. If you are using Windows or Mac OS X, 
then you should also adjust the path in `maddpg.py`.

To install Python dependencies you can simply run `pip install -r requirements.txt` from the parent directory.

### Instructions
To train the agent uncomment following lines in `maddpg.py`:

``
algo.train()
``

To evaluate the agent uncomment following lines in `maddpg.py` and specify checkpoint:

``
algo.eval(r'./episode1471_score2.02.weights',
              num_episodes=100,
              step_delay=None)
``

You can specify number of episodes to run with `num_episodes`, 
and whether to delay execution of each step with `step_delay`.

### Demo

![gif](https://thumbs.gfycat.com/BelovedImpressionableFairybluebird-size_restricted.gif)
