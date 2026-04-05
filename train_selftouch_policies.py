"""
Train three PPO policies for Self-Touch using an "active touch sensors" intrinsic reward.

  Policy A:   1,000 timesteps
  Policy B:  10,000 timesteps
  Policy C: 100,000 timesteps

For each policy this script produces:
  • A reward-curve plot  (results/self_touch/policy_<X>_reward_curve.png)
  • A 30-second MP4 video with a side panel that shows, in real-time,
    (i)  a 3-D scatter view of every touch sensor (active = red),
    (ii) a bar chart of the per-body-part activation fraction.
    (results/self_touch/videos/policy_<X>_behavior.mp4)
"""

import os
import sys
import yaml
import numpy as np

# ── set non-interactive backend BEFORE any pyplot import ──────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(".")
sys.path.append("..")
import mimoEnv                          # registers BabyBench gym envs
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils


# ══════════════════════════════════════════════════════════════════════════════
# Wrapper – intrinsic reward = fraction of active touch sensors
# ══════════════════════════════════════════════════════════════════════════════

class ActiveTouchWrapper(gym.Wrapper):
    """Reward: fraction of touch-sensor components whose value exceeds threshold."""

    def compute_intrinsic_reward(self, obs):
        return float(np.sum(obs["touch"] > 1e-6) / len(obs["touch"]))

    def step(self, action):
        obs, ext_reward, terminated, truncated, info = self.env.step(action)
        reward = self.compute_intrinsic_reward(obs) + ext_reward  # ext always 0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Callback – record (timestep, episode_reward) pairs during training
# ══════════════════════════════════════════════════════════════════════════════

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self._ep_reward = 0.0
        self.records = []   # list of (global_timestep, episode_reward)

    def _on_step(self):
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.records.append((self.num_timesteps, self._ep_reward))
            self._ep_reward = 0.0
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ══════════════════════════════════════════════════════════════════════════════

def _touch_bar_image(env_inner, width=240, height=240):
    """
    240×240 bar chart: per-body-part fraction of active touch sensors.
    Active sensors are coloured red; idle sensors are dark grey.
    """
    labels, fracs = [], []

    if hasattr(env_inner, "touch") and hasattr(env_inner.touch, "sensor_outputs"):
        for body_id, force_vecs in env_inner.touch.sensor_outputs.items():
            mag = np.linalg.norm(force_vecs, axis=-1)
            fracs.append(float(np.mean(mag > 1e-7)))
            try:
                raw = env_inner.model.body(body_id).name
                # shorten common prefixes for readability
                name = (raw.replace("mimo_", "")
                           .replace("right_", "R.")
                           .replace("left_",  "L."))
            except Exception:
                name = f"B{body_id}"
            labels.append(name)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    if fracs:
        x = np.arange(len(fracs))
        colors = ["#ff3333" if v > 0.005 else "#444444" for v in fracs]
        ax.bar(x, fracs, color=colors, width=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=5, color="white")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", ".5", "1"], fontsize=6, color="white")
    else:
        ax.text(0.5, 0.5, "no touch data", ha="center", va="center",
                transform=ax.transAxes, color="white", fontsize=8)

    ax.set_title("Touch Activations", fontsize=8, color="white", pad=2)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555555")
    ax.tick_params(colors="white", length=2)

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]   # RGBA → RGB
    plt.close(fig)
    return cv2.resize(buf, (width, height))


def _build_frame(wrapped_env):
    """
    Compose a 480×720 RGB frame:
      left  (480×480): corner camera view
      upper-right (240×240): 3-D touch-sensor scatter (active = red)
      lower-right (240×240): per-body activation bar chart
    """
    inner = wrapped_env.env            # unwrapped BabyBench env

    # --- corner view ----------------------------------------------------------
    corner = bb_utils.render(inner, "corner")           # (render_size, render_size, 3)
    if corner.shape[0] != 480 or corner.shape[1] != 480:
        corner = cv2.resize(corner, (480, 480))

    # --- 3-D touch scatter ----------------------------------------------------
    touch_3d = bb_utils.view_touches(inner, focus_body="hip",
                                     contact_with="hands")    # 240×240

    # --- activation bar chart -------------------------------------------------
    touch_bar = _touch_bar_image(inner, width=240, height=240)

    # --- compose --------------------------------------------------------------
    frame = np.zeros((480, 720, 3), dtype=np.uint8)
    frame[:, :480]   = corner
    frame[:240, 480:] = touch_3d
    frame[240:, 480:] = touch_bar
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Reward-curve plot
# ══════════════════════════════════════════════════════════════════════════════

def _plot_rewards(callback, policy_name, total_timesteps, save_dir):
    records = callback.records
    path = os.path.join(save_dir, f"policy_{policy_name}_reward_curve.png")

    fig, ax = plt.subplots(figsize=(10, 4))

    if not records:
        # Fewer than 1 episode completed (e.g. Policy A with 1 K steps)
        ax.text(0.5, 0.5,
                f"Training ran for {total_timesteps:,} steps\n"
                f"(episode not completed – reward data unavailable)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#555555")
    else:
        ts   = [r[0] for r in records]
        rews = [r[1] for r in records]

        ax.plot(ts, rews, alpha=0.35, color="steelblue",
                linewidth=0.8, label="episode reward")

        # rolling average (window = ~20 % of episodes, min 2)
        n = len(rews)
        w = max(2, n // 5)
        if n >= w:
            smooth    = np.convolve(rews, np.ones(w) / w, mode="valid")
            smooth_ts = ts[w - 1:]
            ax.plot(smooth_ts, smooth, color="steelblue", linewidth=2.0,
                    label=f"rolling avg (w={w})")

        ax.set_xlabel("Training Timesteps", fontsize=12)
        ax.set_ylabel("Episode Reward\n(active-touch fraction)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    ax.set_title(
        f"Policy {policy_name} – Reward During Training ({total_timesteps:,} steps)",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{policy_name}] reward plot  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 30-second behaviour video
# ══════════════════════════════════════════════════════════════════════════════

def _generate_video(wrapped_env, model, policy_name, save_dir,
                    fps=30, duration_sec=30):
    n_frames = fps * duration_sec
    frames   = []

    obs, _ = wrapped_env.reset()
    for i in range(n_frames):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = wrapped_env.step(action)
        frames.append(_build_frame(wrapped_env))

        if terminated or truncated:
            obs, _ = wrapped_env.reset()

        if (i + 1) % 150 == 0:
            print(f"    [{policy_name}] {i+1}/{n_frames} frames …")

    path = os.path.join(save_dir, "videos", f"policy_{policy_name}_behavior.mp4")
    bb_utils.evaluation_video(frames, path, frame_rate=fps, resolution=(720, 480))
    print(f"  [{policy_name}] video        → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-policy training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def train_and_visualize(policy_name, total_timesteps, config):
    print(f"\n{'='*60}")
    print(f"  Policy {policy_name}  |  {total_timesteps:,} timesteps")
    print(f"{'='*60}")

    save_dir = config["save_dir"]

    # --- build env & wrap -----------------------------------------------------
    env     = bb_utils.make_env(config, training=True)
    wrapped = ActiveTouchWrapper(env)
    wrapped.reset()

    # --- train ----------------------------------------------------------------
    callback = RewardLogger()
    model    = PPO("MultiInputPolicy", wrapped, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model_path = os.path.join(save_dir, f"model_{policy_name}")
    model.save(model_path)
    print(f"  [{policy_name}] model        → {model_path}")

    # --- reward plot ----------------------------------------------------------
    _plot_rewards(callback, policy_name, total_timesteps, save_dir)

    # --- 30-second video ------------------------------------------------------
    print(f"  [{policy_name}] Rendering 30-second video ({30*30} frames) …")
    _generate_video(wrapped, model, policy_name, save_dir)

    wrapped.close()
    print(f"  [{policy_name}] done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    config_path = "examples/config_selftouch.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    policies = [
        ("A",   1_000),
        ("B",  10_000),
        ("C", 100_000),
    ]

    for name, steps in policies:
        train_and_visualize(name, steps, config)

    print(f"\nAll done!  Results saved to: {config['save_dir']}/")
    print("  Reward plots : policy_A/B/C_reward_curve.png")
    print("  Videos       : videos/policy_A/B/C_behavior.mp4")


if __name__ == "__main__":
    main()
