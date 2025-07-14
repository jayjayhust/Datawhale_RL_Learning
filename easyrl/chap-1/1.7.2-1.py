import gym
env = gym.make('MountainCar-v0', render_mode='human')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low,
        env.observation_space.high))  # 打印观测空间的上下限
print('动作数 = {}'.format(env.action_space.n))

class SimpleAgent:
    def __init__(self, env):
        pass
    
    def decide(self, observation):  # 决策
        position, velocity = observation  # 提取位置和速度
        # 计算下坡的速度下限
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,  # 一个二次函数，开口向下，顶点在 position = -0.25 处
                0.3 * (position + 0.9) ** 4 - 0.008)  # 一个四次函数，形状更复杂，但整体趋势是随着 position 的增加先增后减
        # 计算下坡的速度上限
        ub = -0.07 * (position + 0.38) ** 2 + 0.07  # 这是一个二次函数，开口向下，顶点在 position = -0.38 处
        if lb < velocity < ub:
            action = 2  # 执行加速
        else:
            action = 0  # 执行减速
        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass
    
agent = SimpleAgent(env)

def play(env, agent, render=False, train=False):
    episode_reward = 0.  # 记录回合总奖励，初始值为0
    observation = env.reset()  # 重置游戏环境，开始新回合
    while True:  # 不断循环，直到回合结束
        if render:  # 判断是否显示
            env.render()  # 显示图形界面
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)  # 执行动作
        episode_reward += reward # 收集回合奖励
        if train:  # 判断是否训练智能体
            agent.learn(observation, action, reward, done)  # 学习
        if done:  # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward  # 返回回合总奖励

observation = env.reset(seed=3)  # 设置随机种子，让结果可复现

episode_reward = play(env, agent, render=False)
print('回合奖励 = {}'.format(episode_reward))
# episode_rewards = [play(env, agent) for _ in range(100)]
# print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))

env.close()  # 关闭图形界面
