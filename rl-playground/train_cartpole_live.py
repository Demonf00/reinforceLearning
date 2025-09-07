# train_cartpole_live.py
import warnings, math, time
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning, module="pygame.pkgdata")

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from PIL import Image, ImageDraw, ImageFont
import cv2

# ==== 文本绘制：兼容 Pillow 新旧版本 ====
try:
    FONT = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 18)
except Exception:
    try:
        FONT = ImageFont.truetype("consola.ttf", 18)
    except Exception:
        FONT = ImageFont.load_default()

def _ml_text_size(draw, text, font, spacing):
    if hasattr(draw, "multiline_textbbox"):
        l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="left")
        return (r - l), (b - t)
    # 旧 Pillow 兜底
    lines = text.splitlines() or [text]
    widths, heights = zip(*(draw.textsize(line, font=font) for line in lines))
    return max(widths), sum(heights) + (len(lines) - 1) * spacing

def annotate(frame: np.ndarray, lines, step_idx: int, ep_steps: int):
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    pad = 8
    text = "\n".join(lines)
    tw, th = _ml_text_size(draw, text, font=FONT, spacing=4)
    draw.rectangle([0, 0, tw + 2*pad, th + 2*pad], fill=(0,0,0,140))
    draw.multiline_text((pad, pad), text, font=FONT, fill=(255,255,255,230), spacing=4)

    W, H = img.size
    bar_h = 12
    draw.rectangle([0, H - bar_h, W, H], fill=(0,0,0,120))
    if ep_steps > 1:
        ratio = max(0.0, min(1.0, step_idx / (ep_steps - 1)))
        draw.rectangle([0, H - bar_h, int(W * ratio), H], fill=(255,255,255,200))

    out = Image.alpha_composite(img, overlay).convert("RGB")
    return np.asarray(out)

def cartpole_fail_reason(env_unwrapped, obs, done, info):
    if not done: return ""
    if info.get("TimeLimit.truncated", False):
        return "END: 达到最大步数（时间上限）"
    x, x_dot, theta, theta_dot = obs
    x_thr = getattr(env_unwrapped, "x_threshold", 2.4)
    th_thr = getattr(env_unwrapped, "theta_threshold_radians", 12 * math.pi / 180.0)
    if abs(x) > x_thr:
        return f"FAIL: |x|={abs(x):.2f} > {x_thr:.2f} (跑出轨道)"
    if abs(theta) > th_thr:
        return f"FAIL: |θ|={abs(theta)*180/math.pi:.1f}° > {th_thr*180/math.pi:.1f}° (杆倒)"
    return "FAIL: 终止（原因未知）"

class LiveViewerCallback(BaseCallback):
    """
    训练时实时显示：
      - 动作/奖励/累计回报/状态(位置/角度)/策略概率 π(a|s)/价值 V(s)
      - 失败原因（跑出轨道/杆倒/时间上限）
    热键：
      - q：立刻停止训练
      - p：暂停/继续可视化（训练不停，只是停更新图像）
    """
    def __init__(self, render_every=1, window_name="RL Live", save_video=None, fps=30):
        super().__init__()
        self.render_every = render_every
        self.window_name = window_name
        self.save_video = save_video
        self.fps = fps
        self.paused = False

        self.ep_return = 0.0
        self.ep_steps = 0
        self.frames = []  # 可选：保存视频

    def _on_training_start(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def _on_step(self) -> bool:
        # 从 locals 里拿到当前 transition 信息（SB3 提供）
        obs = np.array(self.locals["new_obs"])[0]          # (4,)
        action = int(np.array(self.locals["actions"])[0])  # 0/1
        reward = float(np.array(self.locals["rewards"])[0])
        done = bool(np.array(self.locals["dones"])[0])
        info = self.locals["infos"][0]

        self.ep_return += reward

        # 计算策略概率与状态价值（用于叠加显示）
        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        dist = self.model.policy.get_distribution(obs_t)
        if hasattr(dist.distribution, "probs"):
            probs = dist.distribution.probs.detach().cpu().numpy().squeeze()
        else:
            logits = dist.distribution.logits
            probs = logits.softmax(dim=-1).detach().cpu().numpy().squeeze()
        value = self.model.policy.predict_values(obs_t).detach().cpu().numpy().squeeze()

        # 渲染图像（DummyVecEnv.render 返回 numpy 图；1 个 env 就是一张图）
        if (self.n_calls % self.render_every == 0) and (not self.paused):
            img = self.training_env.render(mode="rgb_array")
            frame = img if isinstance(img, np.ndarray) else img[0]

            x, x_dot, theta, theta_dot = obs
            lines = [
                f"Global Step: {self.num_timesteps:,}",
                f"Episode Step: {self.ep_steps}   Reward: {reward:+.2f}   Return: {self.ep_return:.2f}",
                f"Action: {action} ({'LEFT' if action==0 else 'RIGHT'})",
                f"x={x:+.3f} x_dot={x_dot:+.3f}  θ={theta*180/math.pi:+.2f}°  θ_dot={theta_dot:+.3f}",
                f"π(a|s)={np.array2string(probs, precision=3)}    V(s)={value:.3f}",
            ]
            reason = cartpole_fail_reason(self.training_env.envs[0].unwrapped, obs, done, info)
            if reason: lines.append(reason)

            ann = annotate(frame, lines, self.ep_steps, max(1, self.ep_steps+1))
            cv2.imshow(self.window_name, ann[..., ::-1])  # RGB->BGR

            if self.save_video is not None:
                self.frames.append(ann)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False  # 终止训练
            elif key == ord('p'):
                self.paused = not self.paused

        # 处理回合边界
        if done:
            # 回合结束可在窗口标题显示一下
            cv2.setWindowTitle(self.window_name, f"{self.window_name} | EpReturn={self.ep_return:.2f}  steps={self.ep_steps}")
            self.ep_return = 0.0
            self.ep_steps = 0
        else:
            self.ep_steps += 1

        return True

    def _on_training_end(self) -> None:
        if self.save_video and len(self.frames):
            import imageio
            imageio.mimsave(self.save_video, self.frames, fps=self.fps)
            print(f"[Saved] {self.save_video}")
        cv2.destroyAllWindows()

def make_env():
    # 关键：render_mode="rgb_array" 才能在训练时取到帧
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return Monitor(env)

if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, tensorboard_log="./tb", verbose=1)

    cb = LiveViewerCallback(
        render_every=1,             # 每步渲染；想更快可改为 2/4/10
        window_name="CartPole Live",
        save_video="train_live.mp4",# 训练过程同时录一份视频；不要就设为 None
        fps=30
    )

    model.learn(total_timesteps=100_000, callback=cb)
    model.save("ppo_cartpole_live")
    print("训练完成。")
