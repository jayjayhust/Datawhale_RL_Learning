# Task06：DDPG算法、PPO算法、SAC算法、阅读视觉强化学习/ICLR'25Oral论文


## DDPG(Deep Deterministic Policy Gradient)算法
DDPG算法是一种结合了深度学习和确定性策略梯度(DPG)的强化学习算法，主要用于连续动作空间的问题。

首先我们知道DQN算法的一个主要缺点就是不能用于连续动作空间，这是因为在DQN算法中动作是通过贪心策略或者说argmax的方式来从Q函数间接得到（这里Q函数就相当于DDPG算法中的Critic网络，而Critic网络的输出是一个连续值）。想要适配连续的动作空间，就需要用到确定性策略梯度(DPG)算法。DPG核心是将选择动作的过程变成一个直接从状态映射到具体动作的函数，也就是所谓的策略网络。
![μθ(s)](../../images/task06_11-1.png)

DDPG算法的核心思想是使用两个神经网络，一个用于策略（Actor），另一个用于值函数评估（Critic）。在训练过程中，通过最小化Critic网络的损失来更新其参数，同时利用更新的Critic网络来改进Actor网络。

DDPG算法的优势和劣势：
- 优势：
  - 适用于连续动作空间，效果好
  - 高效的梯度优化
  - 经验回放和目标网络
- 劣势：
  - 只适用于连续动作空间
  - 高度依赖超参数
  - 高度敏感的初始条件
  - 容易陷入局部最优

## TD3(Twin Delayed DDPG)算法
DDPG算法的缺点太多明显，因此后来有人对其进行了改进，这就是我们接下来要介绍的TD3算法，中文全称为双延迟确定性策略梯度算法。相对于DDPG算法，TD3算法的改进主要做了三点重要的改进，一是双Q网络，体现在名字中的Twin，二是延迟更新，体现在名字中的Delayed，三是躁声正则（noise regularisation）。

## PPO(Proximal Policy Optimization)算法
PPO是TRPO(Trust Region Policy Optimization)和A2C(Actor-Critic)的结合体，是一种on-policy的算法。PPO算法在相关应用中有着非常重要的地位，是一个里程碑式的算法。不同于DDPG算法，PPO算法是一类典型的Actor-Critic算法，既适用于连续动作空间，也适用于离散动作空间。

PPO算法是一种基于策略梯度的强化学习算法，算法的主要思想是通过在策略梯度的优化过程中引入一个重要性权重来限制策略更新的幅度，从而提高算法的稳定性和收敛性。PPO算法的优点在于简单、易于实现、易于调参，应用十分广泛，正可谓 “遇事不决PPO”。

[RSL_RL项目](https://github.com/leggedrobotics/rsl_rl)，一个经典的在GPU上运行的强化学习库，包含PPO算法实现、向量化环境、Actor-Critic网络结构等组件。

## SAC(Soft Actor-Critic)算法
SAC算法是一种基于最大熵强化学习的策略梯度算法，它的目标是最大化策略的熵，从而使得策略更加鲁棒。SAC算法的核心思想是，通过最大化策略的熵，使得策略更加鲁棒，经过超参改良后的SAC算法在稳定性方面是可以与PPO算法华山论剑的。

## 阅读视觉强化学习/ICLR'25Oral论文

