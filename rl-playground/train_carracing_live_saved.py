# train_carracing_live_safe.py
# CarRacing-v2 连续动作 + CNN + 帧堆叠
# ✅ 异常/中断也会保存 MP4
# ✅ 定期 checkpoint，支持断点续训（含 VecNormalize 统计）
# ✅ 低内存设置（小分辨率、较短 n_steps、视频流式写盘）
import os, atexit, warnings, math
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
try:
    from gymnasium.wrappers import GrayScaleObservation  # 新写法
except Exception:
    from gymnasium.wrappers import GrayscaleObservation as GrayScaleObservation  # 旧写法
from gymnasium import ObservationWrapper, Wrapper, spaces

import torch
import torch.nn as nn
import cv2
import imageio
from PIL import Image, ImageDraw, ImageFont

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecTransposeImage, VecFrameStack, VecNormalize
)
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

# ========= 文本叠加 =========
def _load_font():
    for p in [r"C:\Windows\Fonts\consola.ttf", "consola.ttf"]:
        try: return ImageFont.truetype(p, 18)
        except Exception: pass
    return ImageFont.load_default()
FONT = _load_font()

def _ml_text_size(draw, text, font, spacing):
    if hasattr(draw, "multiline_textbbox"):
        l,t,r,b = draw.multiline_textbbox((0,0), text, font=font, spacing=spacing, align="left")
        return (r-l), (b-t)
    lines = text.splitlines() or [text]
    ws, hs = zip(*(draw.textsize(line, font=font) for line in lines))
    return max(ws), sum(hs) + (len(lines)-1)*spacing

def annotate(frame: np.ndarray, lines, step_idx: int, ep_steps: int):
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    pad = 8; text = "\n".join(lines)
    tw, th = _ml_text_size(d, text, FONT, 4)
    d.rectangle([0,0, tw+2*pad, th+2*pad], fill=(0,0,0,160))
    d.multiline_text((pad,pad), text, font=FONT, fill=(255,255,255,230), spacing=4)
    W,H = img.size; bar_h = 12
    d.rectangle([0, H-bar_h, W, H], fill=(0,0,0,120))
    if ep_steps>1:
        ratio = max(0.0, min(1.0, step_idx/max(1, ep_steps-1)))
        d.rectangle([0, H-bar_h, int(W*ratio), H], fill=(255,255,255,200))
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))

# ========= 观测与动作封装 =========
class EnsureChannelLast(ObservationWrapper):
    """确保 (H,W,1)，避免某些版本掉通道维导致形状不匹配。"""
    def __init__(self, env):
        super().__init__(env)
        s = env.observation_space
        if len(s.shape) == 2:
            h,w = s.shape
            self.observation_space = spaces.Box(0,255,shape=(h,w,1),dtype=s.dtype)
        else:
            self.observation_space = s
    def observation(self, obs):
        return obs[...,None] if obs.ndim==2 else obs

class ActionRepeat(Wrapper):
    """一次动作重复 N 帧，稳定探索、加快推进。"""
    def __init__(self, env, repeat=4): super().__init__(env); self.repeat=repeat
    def step(self, action):
        total_r = 0.0
        for _ in range(self.repeat):
            obs, r, term, trunc, info = self.env.step(action)
            total_r += r
            if term or trunc:
                return obs, total_r, term, trunc, info
        return obs, total_r, term, trunc, info

# ========= 训练时实时可视化（流式写盘，异常也保存） =========
class LiveViewer(BaseCallback):
    """
    热键：
      q：立刻停止训练
      p：暂停/继续画面刷新（训练不停）
    """
    def __init__(self, every=6, win="CarRacing Live",
                 save_video="carracing_train.mp4", fps=30, video_stride=3):
        super().__init__()
        self.every = every
        self.win = win
        self.save_video = save_video
        self.fps = fps
        self.video_stride = video_stride  # 每隔多少步写一帧
        self.paused = False
        self.ep_steps = 0
        self.ep_return = 0.0
        self.writer = None

    def _open_writer(self):
        if self.save_video and self.writer is None:
            self.writer = imageio.get_writer(self.save_video, fps=self.fps)
            atexit.register(self.close)  # 异常/中断也会关闭写入器

    def close(self):
        if self.writer is not None:
            try:
                self.writer.close()
            finally:
                self.writer = None
                print(f"[Saved] {self.save_video}")

    def _on_training_start(self):
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        self._open_writer()
        print(f"[device] {self.model.policy.device} | cuda={torch.cuda.is_available()}")

    def _on_step(self):
        rew  = float(np.array(self.locals["rewards"])[0])
        done = bool(np.array(self.locals["dones"])[0])
        info = self.locals["infos"][0]
        self.ep_return += rew

        # 连续动作策略统计（mean/std）
        obs = np.array(self.locals["new_obs"])[0]
        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        dist = self.model.policy.get_distribution(obs_t).distribution
        base = getattr(dist, "base_dist", dist)
        mean = base.loc.detach().cpu().numpy().squeeze()
        std  = base.scale.detach().cpu().numpy().squeeze()
        action = np.array(self.locals["actions"])[0].astype(float).tolist()

        if (self.n_calls % self.every == 0) and (not self.paused):
            frame = self.training_env.envs[0].unwrapped.render()
            val = self.model.policy.predict_values(obs_t).detach().cpu().numpy().squeeze()
            lines = [
                f"Global: {self.num_timesteps:,} | EpStep: {self.ep_steps}  R:{rew:+.3f}  G:{self.ep_return:.2f}",
                f"Action [steer, gas, brake] = [{action[0]:+0.3f}, {action[1]:0.3f}, {action[2]:0.3f}]",
                f"Policy mean = [{mean[0]:+0.3f}, {mean[1]:0.3f}, {mean[2]:0.3f}]  std = [{std[0]:0.3f}, {std[1]:0.3f}, {std[2]:0.3f}]",
                f"V(s) = {val:.3f}",
            ]
            if done:
                lines.append("TRUNCATED: 时间上限" if info.get("TimeLimit.truncated", False) else "END")
            ann = annotate(frame, lines, self.ep_steps, max(1, self.ep_steps+1))

            # 显示
            cv2.imshow(self.win, ann[..., ::-1])
            # 流式写盘（不占内存）
            if self.writer and (self.num_timesteps % self.video_stride == 0):
                self.writer.append_data(ann)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): return False
            if key == ord('p'): self.paused = not self.paused

        if done:
            cv2.setWindowTitle(self.win, f"{self.win} | Return={self.ep_return:.2f} Steps={self.ep_steps}")
            self.ep_steps = 0
            self.ep_return = 0.0
        else:
            self.ep_steps += 1
        return True

    def _on_training_end(self):
        self.close()
        cv2.destroyAllWindows()

# ========= 环境 =========
def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    env = ActionRepeat(env, repeat=4)            # 更稳探索/更快推进
    env = Monitor(env)
    env = GrayScaleObservation(env, keep_dim=True)   # -> (96,96,1)
    env = ResizeObservation(env, (72, 72))           # ↓ 分辨率，降内存/显存
    env = EnsureChannelLast(env)                     # -> (72,72,1)
    return env

if __name__ == "__main__":
    # ========== 构建/恢复 VecNormalize ==========
    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
    if os.path.exists("vecnorm.pkl"):
        venv = VecNormalize.load("vecnorm.pkl", venv)   # 恢复统计
    venv.training = True

    # 帧堆叠 -> 通道前置（供 CnnPolicy）
    venv = VecFrameStack(venv, n_stack=4, channels_order="last")  # (72,72,4)
    venv = VecTransposeImage(venv)                                 # (4,72,72)

    # ========== 断点续训：加载最新模型（如存在） ==========
    resume_path = "./checkpoints/ppo_carracing_latest.zip"
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256,256], vf=[256,256]),
        log_std_init=0.8,        # 初期更强探索
        ortho_init=False
    )
    if os.path.exists(resume_path):
        model = PPO.load(resume_path, env=venv, device="cuda", tensorboard_log="./tb", verbose=1)
        print("[resume] loaded", resume_path)
    else:
        model = PPO(
            "CnnPolicy",
            venv,
            device="cuda",                 # 无 GPU 可改 "auto"/"cpu"
            verbose=1,
            tensorboard_log="./tb",
            n_steps=1024,                  # 低内存关键参数
            batch_size=256,
            learning_rate=3e-4,
            ent_coef=0.02,
            gamma=0.999,
            gae_lambda=0.95,
            n_epochs=10,
            clip_range=0.1,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs
        )

    # ========== 回调：checkpoint + 评估 + 实时可视化（流式写盘） ==========
    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./checkpoints", name_prefix="ppo_carracing")
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    if os.path.exists("vecnorm.pkl"):
        eval_env = VecNormalize.load("vecnorm.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False  # 评估时还原真实奖励
    eval_cb = EvalCallback(eval_env, eval_freq=100_000, n_eval_episodes=3,
                           best_model_save_path="./models", log_path="./logs")

    live_cb = LiveViewer(every=6, save_video="carracing_train.mp4", fps=30, video_stride=3)

    # ========== 训练（无论异常与否都安全保存） ==========
    try:
        model.learn(total_timesteps=1_000_000,
                    callback=[live_cb, ckpt_cb, eval_cb],
                    reset_num_timesteps=False)  # 断点续训保留步数
    finally:
        # 强制保存最新模型与 VecNormalize 统计
        model.save("./checkpoints/ppo_carracing_latest")
        # 注意：保存的是「训练用」venv 的 VecNormalize
        venv.save("vecnorm.pkl")
        live_cb.close()
        print("[safe-exit] model & vecnorm saved")

    print("训练完成。")