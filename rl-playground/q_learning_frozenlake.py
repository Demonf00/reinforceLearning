# q_learning_frozenlake.py
import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=True)  # 离散格子
nS, nA = env.observation_space.n, env.action_space.n
Q = np.zeros((nS, nA))

episodes = 5000
alpha = 0.8     # 学习率
gamma = 0.95    # 折扣
epsilon = 1.0   # 探索
eps_min = 0.05
eps_decay = 0.999

rewards = []
for ep in range(episodes):
    s, info = env.reset()
    total = 0
    done = False
    while not done:
        a = env.action_space.sample() if random.random() < epsilon else np.argmax(Q[s])
        s2, r, term, trunc, info = env.step(a)
        done = term or trunc
        Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * np.max(Q[s2]))
        s = s2
        total += r
    epsilon = max(eps_min, epsilon * eps_decay)
    rewards.append(total)

print("近100回合平均回报：", np.mean(rewards[-100:]))

# 测试一次并打印策略走法
s, info = env.reset(seed=0)
done = False
steps = 0
while not done and steps < 100:
    a = np.argmax(Q[s])
    s, r, term, trunc, info = env.step(a)
    done = term or trunc
    steps += 1
print("测试步数：", steps)
