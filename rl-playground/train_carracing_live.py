# train_carracing_live.py
# 连续动作（方向盘/油门/刹车）+ CNN + 帧堆叠 + 训练时实时可视化
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
try:
    from gymnasium.wrappers import GrayScaleObservation  # 新写法
except Exception:
    from gymnasium.wrappers import GrayscaleObservation as GrayScaleObservation  # 旧写法

from gymnasium import ObservationWrapper, spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecTransposeImage, VecFrameStack
)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback


from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import math

# ========= 字体 & 叠加工具 =========
def _load_font():
    for path in [r"C:\Windows\Fonts\consola.ttf", "consola.ttf"]:
        try:
            return ImageFont.truetype(path, 18)
        except Exception:
            pass
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

# ========= 保证通道维存在的封装 =========
class EnsureChannelLast(ObservationWrapper):
    """
    某些 gymnasium 版本在 GrayScale+Resize 后会返回 (H,W) 而不是 (H,W,1)。
    这里统一补上通道维，避免与 VecFrameStack 的期望不一致。
    """
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if len(obs_space.shape) == 2:
            h, w = obs_space.shape
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, 1), dtype=obs_space.dtype
            )
        else:
            self.observation_space = obs_space

    def observation(self, obs):
        if obs.ndim == 2:
            obs = np.expand_dims(obs, axis=-1)  # (H,W) -> (H,W,1)
        return obs

# ========= 实时可视化 Callback =========
class LiveViewer(BaseCallback):
    """
    热键：
      q：立刻停止训练
      p：暂停/继续画面刷新（训练不停）
    """
    def __init__(self, every=2, win="CarRacing Live", save_video=None, fps=30):
        super().__init__()
        self.every = every; self.win = win
        self.save_video = save_video; self.fps = fps
        self.paused = False; self.ep_steps = 0; self.ep_return = 0.0
        self.frames = []

    def _on_training_start(self):
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        print(f"[device] {self.model.policy.device} | cuda_available={torch.cuda.is_available()}")

    def _on_step(self) -> bool:
        rew  = float(np.array(self.locals["rewards"])[0])
        done = bool(np.array(self.locals["dones"])[0])
        info = self.locals["infos"][0]
        self.ep_return += rew

        # 连续动作策略分布（steer∈[-1,1], gas/brake∈[0,1]）
        obs = np.array(self.locals["new_obs"])[0]
        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        dist = self.model.policy.get_distribution(obs_t)
        d = dist.distribution
        try:
            mean = d.mean.detach().cpu().numpy().squeeze()
            std  = d.stddev.detach().cpu().numpy().squeeze()
        except Exception:
            base = getattr(d, "base_dist", d)
            mean = base.loc.detach().cpu().numpy().squeeze()
            std  = base.scale.detach().cpu().numpy().squeeze()
        action = np.array(self.locals["actions"])[0].astype(float).tolist()

        if (self.n_calls % self.every == 0) and (not self.paused):
            frame = self.training_env.envs[0].unwrapped.render()  # 原始彩色
            val = self.model.policy.predict_values(obs_t).detach().cpu().numpy().squeeze()
            lines = [
                f"Global: {self.num_timesteps:,} | EpStep: {self.ep_steps}  R: {rew:+.3f}  G: {self.ep_return:.2f}",
                f"Action [steer, gas, brake] = [{action[0]:+0.3f}, {action[1]:0.3f}, {action[2]:0.3f}]",
                f"Policy mean = [{mean[0]:+0.3f}, {mean[1]:0.3f}, {mean[2]:0.3f}]  std = [{std[0]:0.3f}, {std[1]:0.3f}, {std[2]:0.3f}]",
                f"V(s) = {val:.3f}",
            ]
            if done:
                reason = "END"
                if info.get("TimeLimit.truncated", False):
                    reason = "TRUNCATED: 时间上限"
                lines.append(reason)

            ann = annotate(frame, lines, self.ep_steps, max(1, self.ep_steps+1))
            cv2.imshow(self.win, ann[..., ::-1])
            if self.save_video: self.frames.append(ann)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): return False
            if key == ord('p'): self.paused = not self.paused

        if done:
            cv2.setWindowTitle(self.win, f"{self.win} | Return={self.ep_return:.2f} Steps={self.ep_steps}")
            self.ep_steps = 0; self.ep_return = 0.0
        else:
            self.ep_steps += 1
        return True

    def _on_training_end(self):
        if self.save_video and self.frames:
            import imageio
            imageio.mimsave(self.save_video, self.frames, fps=self.fps)
            print(f"[Saved] {self.save_video}")
        cv2.destroyAllWindows()

# ========= 环境构造 =========
def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    env = Monitor(env)
    env = GrayScaleObservation(env, keep_dim=True)  # 理论上 -> (96,96,1)
    env = ResizeObservation(env, (84, 84))          # -> (84,84,1)（个别版本会掉通道维）
    env = EnsureChannelLast(env)                    # 强制保证 (H,W,1)
    return env

def make_vec_env(training: bool):
    venv = DummyVecEnv([make_env])                               # base
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
    venv.training = training                                     # 训练 True，评估 False
    venv = VecFrameStack(venv, n_stack=4, channels_order="last") # (84,84,4)
    venv = VecTransposeImage(venv)                               # (4,84,84)
    return venv


if __name__ == "__main__":
    # 先向量化；然后帧堆叠（channels_order="last"）；最后转为 CHW 供 CnnPolicy
    venv = DummyVecEnv([make_env])                                   # (84,84,1)
    venv = VecFrameStack(venv, n_stack=4, channels_order="last")     # (84,84,4)
    venv = VecTransposeImage(venv)                                   # (4,84,84)

    train_env = make_vec_env(training=True)
    eval_env  = make_vec_env(training=False)


    model = PPO(
        "CnnPolicy",
        train_env,
        device="cuda",                 # 无 GPU 改 "auto"/"cpu"
        verbose=1,
        tensorboard_log="./tb",
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        clip_range=0.2,
        gamma=0.99,
        vf_coef=0.5,
        ent_coef=0.01
    )

    cb = LiveViewer(every=2, win="CarRacing Live", save_video="carracing_train.mp4", fps=30)
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=100_000,            # 评估频率按需改
        n_eval_episodes=5,
        best_model_save_path="./models",
        log_path="./logs",
    )

    model.learn(total_timesteps=300_000_000, callback=[cb,eval_cb])
    model.save("ppo_carracing_cnn")
    print("训练完成。")
