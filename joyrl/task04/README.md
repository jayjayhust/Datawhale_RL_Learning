# Task04：DQN算法、DQN算法进阶

## DQN算法
DQN(Deep Q-Network)是在Q-learning算法的基础上引入了深度神经网络来近似动作价值函数Q(s, a)，从而能够处理高维的状态空间。
Q表中我们描述状态空间的时候一般用的是状态个数，而在神经网络中我们用的是状态的维度作为输入变量的个数（比如二维坐标、三维坐标）。

DQN算法相比于Q-learning算法主要有以下改进：
- 经验回放：首先每次用单个样本去迭代网络参数很容易导致训练的不稳定，从而影响模型的收敛，在深度学习基础的章节中我们也讲过小批量梯度下降是目前比较成熟的方式。而在DQN中，我们会把每次与环境交互得到的样本都存储在一个经验回放池中，然后每次从经验池中随机抽取一批样本来训练网络。
- 目标网络：在DQN算法中还有一个重要的技巧，即使用了一个每隔若干步才更新的目标网络。这个技巧其实借鉴了Double DQN算法中的思路，具体会在下一章展开。目标网络和当前网络结构都是相同的，都用于近似Q值，在实践中每隔若干步才把每步更新的当前网络参数复制给目标网络，这样做的好处是保证训练的稳定，避免值的估计发散。

- 代码实战
  - [DQN](./DQN/DQN_CartPole-v1.ipynb)

## DQN算法进阶
- Double DQN：在DQN算法中，我们使用当前网络来选择动作，然后用目标网络计算Q值。而在Double DQN中，我们用当前网络选择动作，但是用目标网络的输出作为下一个状态的价值估计。DQN和Double DQN以及Nature DQN的时间线为：DQN(2013)->Double DQN(2015)->Nature DQN(2015)。
- Nature DQN：Nature DQN尝试用两个Q网络来减少目标Q值计算和要更新Q网络参数之间的依赖关系。一个当前Q网络Q用来选择动作，更新模型参数，另一个目标Q网络Q'用于计算目标Q值。目标Q网络的网络参数不需要迭代更新，而是每隔一段时间从当前Q网络$Q$复制过来，即延时更新，这样可以减少目标Q值和当前的Q值相关性。
- Dueling Network：在Nature DQN中，作者提出了Dueling网络结构，即在神经网络的最后一层将特征分为两部分，一部分用于估计状态价值函数V(s)，另一部分用于估计优势函数A(s, a)。
- Noisy DQN：在Nature DQN中，提出了Noisy DQN算法，即在网络的参数上加入噪声。
- PER(prioritized experience replay) DQN：在Nature DQN中，提出了PER算法，即在经验回放池中对样本进行优先级排序。
- C51(categorical) DQN：在Nature DQN中，提出了C51算法，即在每个状态动作对上估计一个分布而不是单个值。
- Rainbow DQN：在Nature DQN中，提出了Rainbow算法，即结合了上述所有改进的DQN。

- 代码实战
  - [DoubleDQN](./DoubleDQN/DoubleDQN_CartPole-v1.ipynb)
  - [DuelingDQN](./DuelingDQN/Dueling_CartPole-v1.ipynb)
  - [NoisyDQN](./NoisyDQN/Noisy_CartPole-v1.ipynb)
  - [PERDQN](./PERDQN/PER_CartPole-v1.ipynb)
  - [RainbowDQN](./RainbowDQN/Rainbow_CartPole-v1.ipynb)





