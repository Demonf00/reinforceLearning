# train_lander_live.py
import warnings, math, numpy as np
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from PIL import Image, ImageDraw, ImageFont
import cv2, time

# ---- 字体 ----
try:
    FONT = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 18)
except Exception:
    try: FONT = ImageFont.truetype("consola.ttf", 18)
    except Exception: FONT = ImageFont.load_default()

def _ml_text_size(draw, text, font, spacing):
    if hasattr(draw, "multiline_textbbox"):
        l,t,r,b = draw.multiline_textbbox((0,0), text, font=font, spacing=spacing, align="left")
        return (r-l),(b-t)
    lines = text.splitlines() or [text]
    widths, heights = zip(*(draw.textsize(line, font=font) for line in lines))
    return max(widths), sum(heights) + (len(lines)-1)*spacing

def annotate(frame, lines, step_idx, ep_steps):
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    pad = 8; text = "\n".join(lines)
    tw, th = _ml_text_size(d, text, FONT, 4)
    d.rectangle([0,0, tw+2*pad, th+2*pad], fill=(0,0,0,150))
    d.multiline_text((pad,pad), text, font=FONT, fill=(255,255,255,230), spacing=4)
    W,H = img.size; bar_h = 12
    d.rectangle([0, H-bar_h, W, H], fill=(0,0,0,120))
    if ep_steps>1:
        ratio = max(0,min(1, step_idx/max(1, ep_steps-1)))
        d.rectangle([0, H-bar_h, int(W*ratio), H], fill=(255,255,255,200))
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))

ACTION_NAMES = {0:"NONE", 1:"LEFT", 2:"MAIN", 3:"RIGHT"}

def judge_end_reason(obs, done, info):
    """基于观测与 info 粗判：成功着陆 / 撞毁 / 出界 / 时限"""
    if not done: return ""
    # obs: [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]
    x,y,vx,vy,ang,angv,lc,rc = obs
    # Gym 的 TimeLimit 截断
    if info.get("TimeLimit.truncated", False): return "END: 时间上限"
    # 成功特征：双腿着地 + 速度角度都很小
    if lc>0.5 and rc>0.5 and abs(vx)<0.3 and abs(vy)<0.3 and abs(ang)<0.2:
        return "SUCCESS: 平稳着陆"
    # 粗略判断：角度/速度过大当作撞毁；或超出横向范围
    if abs(ang)>0.6 or abs(vy)>1.5 or abs(x)>1.5:
        return "CRASH/OUT: 角度/速度/位置越界"
    return "END"

class LiveViewer(BaseCallback):
    def __init__(self, every=1, win="Lander Live", save_video=None, fps=30):
        super().__init__(); self.every=every; self.win=win
        self.save_video=save_video; self.fps=fps
        self.ep_steps=0; self.ep_return=0.0; self.frames=[]; self.paused=False
    def _on_training_start(self): cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
    def _on_step(self):
        obs  = np.array(self.locals["new_obs"])[0]
        act  = int(np.array(self.locals["actions"])[0])
        rew  = float(np.array(self.locals["rewards"])[0])
        done = bool(np.array(self.locals["dones"])[0])
        info = self.locals["infos"][0]
        self.ep_return += rew

        # 策略概率 & 价值
        obs_t,_ = self.model.policy.obs_to_tensor(obs)
        dist    = self.model.policy.get_distribution(obs_t)
        probs   = getattr(dist.distribution, "probs", None)
        if probs is None:
            logits = dist.distribution.logits
            probs = logits.softmax(dim=-1)
        probs = probs.detach().cpu().numpy().squeeze()
        value = self.model.policy.predict_values(obs_t).detach().cpu().numpy().squeeze()

        # 渲染
        if (self.n_calls % self.every == 0) and (not self.paused):
            img = self.training_env.render(mode="rgb_array")
            frame = img if isinstance(img, np.ndarray) else img[0]
            x,y,vx,vy,ang,angv,lc,rc = obs
            lines = [
                f"Global: {self.num_timesteps:,} | EpStep: {self.ep_steps}  R: {rew:+.2f}  G: {self.ep_return:.2f}",
                f"Action: {act} ({ACTION_NAMES[act]})   π={np.array2string(probs, precision=3)}   V={value:.3f}",
                f"x={x:+.2f} y={y:+.2f}  vx={vx:+.2f} vy={vy:+.2f}",
                f"θ={ang*180/math.pi:+.1f}°  ω={angv*180/math.pi:+.1f}°/s  Leg(L/R)={int(lc)}/{int(rc)}",
            ]
            reason = judge_end_reason(obs, done, info)
            if done: lines.append(reason)
            ann = annotate(frame, lines, self.ep_steps, max(1, self.ep_steps+1))
            cv2.imshow(self.win, ann[..., ::-1])
            if self.save_video: self.frames.append(ann)
            key = cv2.waitKey(1) & 0xFF
            if key==ord('q'): return False
            if key==ord('p'): self.paused = not self.paused

        if done:
            cv2.setWindowTitle(self.win, f"{self.win} | Ret={self.ep_return:.2f} Steps={self.ep_steps}")
            self.ep_steps = 0; self.ep_return = 0.0
        else:
            self.ep_steps += 1
        return True
    def _on_training_end(self):
        if self.save_video and self.frames:
            import imageio
            imageio.mimsave("lander_train.mp4", self.frames, fps=self.fps)
            print("[Saved] lander_train.mp4")
        cv2.destroyAllWindows()

def make_env():
    return Monitor(gym.make("LunarLander-v3", render_mode="rgb_array"))

if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, tensorboard_log="./tb", verbose=1, device="cuda")
    cb = LiveViewer(every=1, win="Lander Live", save_video="lander_train.mp4", fps=30)
    model.learn(total_timesteps=300_000, callback=cb)
    model.save("ppo_lander_live")
    print("训练完成。")
