# view_cartpole.py
import math, time, argparse
import numpy as np
from typing import List, Dict, Any

from PIL import Image, ImageDraw, ImageFont
import gymnasium as gym
from stable_baselines3 import PPO

# —— 可选：更好看的等宽字体（找不到就用默认）
try:
    FONT = ImageFont.truetype("consola.ttf", 18)  # Windows 常见
except Exception:
    FONT = ImageFont.load_default()

def cartpole_fail_reason(env, obs, terminated, truncated):
    # 依据 CartPole 环境的阈值判断失败原因
    x, x_dot, theta, theta_dot = obs
    x_thr = getattr(env.unwrapped, "x_threshold", 2.4)
    th_thr = getattr(env.unwrapped, "theta_threshold_radians", 12 * math.pi / 180.0)

    if terminated:
        if abs(x) > x_thr:
            return f"FAIL: |x|={abs(x):.2f} > {x_thr:.2f} (跑出轨道)"
        if abs(theta) > th_thr:
            return f"FAIL: |θ|={abs(theta)*180/math.pi:.1f}° > {th_thr*180/math.pi:.1f}° (杆倒)"
        return "FAIL: 终止（原因未知）"
    if truncated:
        # 一般是 TimeLimit 截断
        return "END: 达到最大步数（时间上限）"
    return ""

def _ml_text_size(draw, text, font, spacing):
    # Pillow 10+ 推荐用 multiline_textbbox
    if hasattr(draw, "multiline_textbbox"):
        l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="left")
        return (r - l), (b - t)
    # 旧版 Pillow 兜底
    lines = text.splitlines() or [text]
    widths, heights = zip(*(draw.textsize(line, font=font) for line in lines))
    return max(widths), sum(heights) + (len(lines) - 1) * spacing

def annotate(frame: np.ndarray, lines: List[str], step_idx: int, total_steps: int) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pad = 8
    text = "\n".join(lines)
    tw, th = _ml_text_size(draw, text, font=FONT, spacing=4)

    # 顶部信息背景条
    draw.rectangle([0, 0, tw + 2*pad, th + 2*pad], fill=(0, 0, 0, 140))
    draw.multiline_text((pad, pad), text, font=FONT, fill=(255, 255, 255, 230), spacing=4)

    # 底部进度条
    W, H = img.size
    bar_h = 12
    draw.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0, 120))
    if total_steps > 1:
        ratio = max(0.0, min(1.0, step_idx / (total_steps - 1)))
        draw.rectangle([0, H - bar_h, int(W * ratio), H], fill=(255, 255, 255, 200))

    out = Image.alpha_composite(img, overlay).convert("RGB")
    return np.asarray(out)


def rollout_with_overlay(model_path: str, seed: int = 0):
    # 用 rgb_array 渲染拿到逐帧画面
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    model = PPO.load(model_path)

    obs, info = env.reset(seed=seed)
    frames, logs = [], []

    done = False
    step = 0
    while not done and step < 10_000:
        # 计算策略概率与状态价值（仅离散动作的 PPO）
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor)
        if hasattr(dist.distribution, "probs"):
            probs = dist.distribution.probs.detach().cpu().numpy().squeeze()
        else:
            # 兜底：从 logits 软最大
            logits = dist.distribution.logits
            probs = logits.softmax(dim=-1).detach().cpu().numpy().squeeze()
        value = model.policy.predict_values(obs_tensor).detach().cpu().numpy().squeeze()

        action, _ = model.predict(obs, deterministic=True)
        nobs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()

        x, x_dot, theta, theta_dot = obs
        lines = [
            f"Step: {step}",
            f"Action: {action} ({'LEFT' if int(action)==0 else 'RIGHT'})",
            f"Reward: {reward:+.2f}   Return: {sum(l['reward'] for l in logs)+reward:.2f}",
            f"x={x:+.3f}  x_dot={x_dot:+.3f}  θ={theta*180/math.pi:+.2f}°  θ_dot={theta_dot:+.3f}",
            f"π(a|s): {probs}   V(s)={value:.3f}",
        ]
        reason = cartpole_fail_reason(env, nobs, terminated, truncated)
        if reason:
            lines.append(reason)

        frames.append(annotate(frame, lines, step, step+1))  # 先按 step+1 防止 0 除
        logs.append({
            "step": step, "action": int(action), "reward": float(reward),
            "obs": [float(x), float(x_dot), float(theta), float(theta_dot)],
            "probs": probs.tolist(), "value": float(value),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "reason": reason
        })
        obs = nobs
        done = terminated or truncated
        step += 1

    env.close()
    return frames, logs

def save_video(frames: List[np.ndarray], path="episode.mp4", fps=30):
    import imageio
    imageio.mimsave(path, frames, fps=fps)
    print(f"[Saved] {path}")

def save_csv(logs: List[Dict[str, Any]], path="episode.csv"):
    import csv
    keys = ["step","action","reward","terminated","truncated","reason","obs","probs","value"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in logs:
            w.writerow(row)
    print(f"[Saved] {path}")

def launch_ui(frames: List[np.ndarray], logs: List[Dict[str, Any]]):
    import gradio as gr

    N = len(frames)

    with gr.Blocks(title="CartPole Step Viewer") as demo:
        gr.Markdown("### CartPole 逐步回放（动作/奖励/状态/失败原因）")
        idx = gr.Slider(0, max(0, N-1), value=0, step=1, label="Step")
        img = gr.Image(type="numpy", interactive=False)

        with gr.Row():
            btn_prev = gr.Button("◀ 上一步")
            btn_next = gr.Button("下一步 ▶")
            btn_save = gr.Button("导出 MP4 & CSV")

        def show(i):
            i = int(max(0, min(N-1, i)))
            return frames[i]

        def prev(i): return max(0, i-1)
        def next_(i): return min(N-1, i+1)

        def do_save():
            ts = int(time.time())
            mp4 = f"episode_{ts}.mp4"
            csv = f"episode_{ts}.csv"
            save_video(frames, mp4)
            save_csv(logs, csv)
            return gr.update(), f"已导出：{mp4} / {csv}"

        idx.change(fn=show, inputs=idx, outputs=img)
        btn_prev.click(fn=prev, inputs=idx, outputs=idx).then(fn=show, inputs=idx, outputs=img)
        btn_next.click(fn=next_, inputs=idx, outputs=idx).then(fn=show, inputs=idx, outputs=img)

        info = gr.Markdown("")
        btn_save.click(fn=do_save, outputs=[img, info])

        # 初始显示
        img.value = frames[0] if N else None

    demo.launch()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_cartpole.zip", help="已训练模型路径")
    ap.add_argument("--save", action="store_true", help="仅导出 MP4/CSV 不开 UI")
    args = ap.parse_args()

    frames, logs = rollout_with_overlay(args.model)
    if args.save:
        save_video(frames, "episode.mp4", fps=30)
        save_csv(logs, "episode.csv")
    else:
        launch_ui(frames, logs)
