# Task01：绪论、马尔可夫过程、动态规划

## [绪论](https://datawhalechina.github.io/joyrl-book/#/ch1/main)

## [马尔可夫过程](https://datawhalechina.github.io/joyrl-book/#/ch2/main)
- 回报公式：Gt = R(t+1) + γG(t+1) = R(t+1) + γR(t+2) + γ^2G(t+2) + ...

## [动态规划](https://datawhalechina.github.io/joyrl-book/#/ch3/main)
- 动态规划基础编程思想：[路径之和](https://leetcode.cn/problems/unique-paths/solutions/514311/bu-tong-lu-jing-by-leetcode-solution-hzjf/)
- 状态价值函数：Vpi(s) = Epi[Gt|St=s]
- 动作价值函数：Qpi(s,a) = Epi[Gt|St=s,At=a]
- 对应的，状态价值函数和动作价值函数的关系为：Vpi(s) = Qpi(s,a1)*P(a1|s) + Qpi(s,a2)*P(a2|s) + ...
  - P(a*|s)：可以理解为策略函数，一般指在状态s下执行动作a*的概率分布。a*归属于动作空间A。
- 贝尔曼方程：
  - 对于状态价值函数（状态价值函数贝尔曼方程）：
  ![推导过程](../../images/task01_3-1.png)
  ![核心步骤解析](../../images/task01_3-2.png)
  - 对于动作价值函数（动作价值函数贝尔曼方程）：
  ![结果公式](../../images/task01_3-3.png)
  ![推导过程](../../images/task01_3-4.png)
  - 贝尔曼最优方程（Bellman optimality equation）：
  ![结果公式](../../images/task01_3-5.png)
- 策略迭代（Policy Iteration）和价值迭代（Value Iteration）：
  - [带两种逐步推导的例子：公主的营救](https://mp.weixin.qq.com/s/ub4EpRZAtny2KTeJqNimbQ)
  - [GridWorld: Dynamic Programming Demo](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)
    - 策略迭代算法：一次迭代包括两个步骤，首先是策略评估对应Policy Evaluation(one sweep)，然后是策略改进对应Policy Update
    - 价值迭代算法：对应点击Toggle Value Iteration（全流程自动执行）





