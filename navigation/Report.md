# Udacity Deep Reinforcement Learning Nanodegree
## Project 1: Navigation - report

### Solution
During this project DQN algorithm was implemented with following techniques:
* Double DQN
* non-prioritized experience replay
* soft updates
* delayed start of optimization
* interval between consecutive optimization steps

### Performance
The agent achieved average score of +15.58 over 100 consecutive episodes after 627 episodes
of training.

![mean reward](https://imgur.com/kjaQQpr.png)

This score can be confirmed by running evaluation mode.

### Comments
Double DQN with simple feed-forward neural network as a Q-function approximator was sufficient
to solve this task. The neural network used consists of 2 hidden layers, each of 128 neurons.
Rectified Linear Units have been used as activation functions (except for the output layer).
This neural network was trained using 64 transitions sampled from non-prioritized replay memory
at each optimization step. To update target network, soft updates were used with tau equal
to 0.001. Network was optimized only every 4th step. Also, first optimization took place after
64 transitions were collected in replay memory. The replay memory was of size 100000. Finally,
epsilon function used to generate epsilon values is shown below:

![epsilon function](https://imgur.com/EGbhbsd.png)

Disclaimer: the code skeleton used in this projects is based on solution provided in 
[PyTorch's official documentation](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

### Ideas for Future Work
* Implement Dueling Architecture
* Add prioritizing in replay memory
* Parallelize training with [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)
* Implement n-step bootstrapping and/or other techniques described in [Rainbow](https://arxiv.org/abs/1710.02298)
