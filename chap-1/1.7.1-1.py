# pip install gym==0.25.2  # 由于 Gym 库 0.26.0 及其之后的版本对之前的代码不兼容，所以我们安装 0.26.0 之前的 Gym，比如 0.25.2
# pip install pygame  # 为了显示图形界面，我们还需要安装 pygame 库

from gym import envs
env_specs = envs.registry.all()
envs_ids = [env_spec.id for env_spec in env_specs]
print(envs_ids)  # 打印所有可用的环境


import gym  # 导入 Gym 的 Python 接口环境包
env = gym.make('CartPole-v0')  # 构建实验环境
env.reset()  # 重置一个回合
for _ in range(1000):
    env.render()  # 显示图形界面
    action = env.action_space.sample()  # 从动作空间中随机选取一个动作
    observation, reward, done, info = env.step(action)  # 用于提交动作，括号内是具体的动作
    print(observation + ', ' + reward)
env.close()  # 关闭环境

