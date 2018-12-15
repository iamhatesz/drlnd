# Udacity Deep Reinforcement Learning Nanodegree
## Project 2: Continuous Control - report

### Solution
During this project DDPG algorithm was implemented with following techniques:
* non-prioritized experience replay
* soft updates
* delayed start of optimization
* interval between consecutive optimization steps
* number of optimization steps in a single control step

### Performance
The agent achieved average score of +39.14 over 100 consecutive episodes after 1258 episodes
of training.

![mean reward](https://imgur.com/JfiUlCt.png)

This score can be confirmed by running evaluation mode.

### Comments

DDPGG with two simple feed-forward neural networks as a deterministic policy function (Actor)
and Q-function approximator (Critic) was sufficient to solve this task.

Actor consists of one hidden layer with 256 neurons with ReLU activations. In the output layer
tanh was used to keep values in (-1; 1) range.

Critic consists of three hidden layers with 256, 256 and 128 neurons respectively. After first
hidden layer, the action tensor is concatenated. For the activation functions Leaky ReLUs were used.
In the output layer no activation function was used.

These neural networks were trained with respect to DDPG algorithm, using a batch of 128 experiences
in each optimization step. Optimization was being done two times (every time with different batch)
every two time steps. This allowed to introduce stability, without increasing the training time significantly.
The replay memory used was non-prioritized replay memory of size 2^20. The optimization process started
after gaining 2^13 samples in it. Both actor and critic had target networks, which were softly updated
with tau equal to 0.01.

Finally, to allow exploration, the noise was added to the action space. The noise used in this projects was a realization
of Ornstein-Uhlenbeck process with initial mean equal 1.0, final mean 0.0, theta 0.02, and sigma 1.0.
For each element of the action space, the separate noise was generated. Below you can find an example of noise
added to the first element of the action vector. Note, that after 1,000,000 of steps, the noise was no longer being added.

![noise function](https://imgur.com/ThSRjFZ.png)

### Ideas for Future Work
* Add prioritizing in replay memory
* Parallelize training with more workers collecting experiences
