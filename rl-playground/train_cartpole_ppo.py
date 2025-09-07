# train_cartpole_ppo.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import imageio
import torch

def make_env():
    env = gym.make("CartPole-v1")  # render_mode=None（训练更快）
    env = Monitor(env)             # 记录回报，便于可视化
    return env

# === 1) 训练 ===
env = DummyVecEnv([make_env])
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb",  # TensorBoard 日志
    device="cuda" if torch.cuda.is_available() else "cpu",
)
eval_env = gym.make("CartPole-v1")
eval_cb = EvalCallback(
    eval_env, best_model_save_path="./models",
    log_path="./logs", eval_freq=5000, n_eval_episodes=5
)
model.learn(total_timesteps=100_000, callback=eval_cb)
model.save("ppo_cartpole")

# === 2) 评估并录制视频 ===
video_env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = video_env.reset(seed=0)
frames = []
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = video_env.step(action)
    frames.append(video_env.render())
    if terminated or truncated:
        break
imageio.mimsave("cartpole.mp4", frames, fps=30)
print("视频已保存为 cartpole.mp4。")
