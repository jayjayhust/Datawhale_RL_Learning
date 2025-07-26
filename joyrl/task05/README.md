# Task05：策略梯度、Actor-critic算法

## 策略梯度算法
策略梯度（policy-based）的算法，与前面介绍的基于价值（value-based）的算法（包括DQN等算法）不同，这类算法直接对策略本身进行近似优化。

以DQN算法为代表的基于价值的算法在很多任务上都取得了不错的效果，并且具备较好的收敛性，但是这类算法也存在一些缺点。
- 无法表示连续动作（比如前面的CartPole-v1是输出的左、右动作）。由于DQN等算法是通过学习状态和动作的价值函数来间接指导策略的，因此它们只能处理离散动作空间的问题，无法表示连续动作空间的问题。而在一些问题中，比如机器人的运动控制问题，连续动作空间是非常常见的，比如要控制机器人的运动速度、角度等等，这些都是连续的量。
<details>
    <summary> 点击时的区域标题 </summary>
    ```python
    class Policy:
        ...   
        def sample_action(self, state):
            ''' 采样动作
            '''
            self.sample_count += 1
            # epsilon指数衰减
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                math.exp(-1. * self.sample_count / self.epsilon_decay) 
            if random.random() > self.epsilon:
                with torch.no_grad():
                    state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                    q_values = self.policy_net(state)
                    action = q_values.max(1)[1].item()  # choose action corresponding to the maximum q value
            else:
                action = random.randrange(self.action_dim)
            return action
        
        @torch.no_grad()  # 不计算梯度，该装饰器效果等同于with torch.no_grad()：
        def predict_action(self, state):
            ''' 预测动作
            '''
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()  # choose action corresponding to the maximum q value
            return action
    ```
</details>
- 高方差。基于价值的方法通常都是通过采样的方式来估计价值函数，这样会导致估计的方差很高，从而影响算法的收敛性。尽管一些DQN改进算法，通过改善经验回放、目标网络等方式，可以在一定程度上减小方差，但是这些方法并不能完全解决这个问题。
- 探索与利用的平衡问题。DQN等算法在实现时通常选择贪心的确定性策略，而很多问题的最优策略是随机策略，即需要以不同的概率选择不同的动作。虽然可以通过DQN策略等方式来实现一定程度的随机策略，但是实际上这种方式并不是很理想，因为它并不能很好地平衡探索与利用的关系。

## Actor-critic算法

