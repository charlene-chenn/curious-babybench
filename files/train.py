"""
train.py — Main training loop for the self-touch world model project.

=== WHAT THIS IS ===
The script you actually run. It ties together:
  - BabyBench environment (MIMo self-touch)
  - World model (learns body representation)
  - RND (generates curiosity rewards)
  - PPO agent (learns touch policy with dual value heads)

=== HOW TO RUN ===
Single experiment:
    python train.py --alpha 0.5 --episodes 500 --seed 42

Full sweep:
    for alpha in 0.0 0.25 0.5 0.75 1.0; do
        for seed in 1 2 3; do
            python train.py --alpha $alpha --seed $seed
        done
    done

=== FRONTIER CONTEXT ===
The training loop follows the standard model-based RL pattern from DreamerV3:
  1. ACT in the real environment → collect experience
  2. LEARN the world model from real experience
  3. COMPUTE intrinsic rewards (RND)
  4. UPDATE the policy (PPO with combined rewards)
  5. REPEAT

The key difference from pure DreamerV3: we train the policy on REAL
experience (not imagined trajectories), but use the world model quality
as a metric for how well the agent understands its body. This simplification
is appropriate for a coursework project and still captures the core ideas.
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import yaml
from collections import defaultdict
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_MIMO_ROOT = os.path.join(_REPO_ROOT, "MIMo")
for _path in (_HERE, _REPO_ROOT, _MIMO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils

from config import Config, get_config
from world_model import WorldModel
from rnd import RND
from agent import PPOAgent

# ═══════════════════════════════════════════════════════════════════
# BABYBENCH ENVIRONMENT WRAPPER
# ═══════════════════════════════════════════════════════════════════

class BabyBenchWrapper:
    """
    Wrapper around BabyBench's self-touch environment.

    === INTEGRATION NOTE ===
    BabyBench uses MuJoCo (via MIMo) not Gymnasium. The API is:
        env = babybench.make("self_touch", ...)
        obs = env.reset()
        obs, info = env.step(action)

    The 'info' dict contains touch sensor data that we use for:
      1. Computing extrinsic reward (number of self-touch contacts)
      2. Tracking which body parts were touched (for your analysis)

    If you haven't installed BabyBench yet, this wrapper falls back to
    a simple simulation that mimics the API for testing your code.
    """

    # Map Config.env_name to the corresponding YAML config file
    _YAML_CONFIGS = {
        "self_touch":  "config_selftouch.yml",
    }

    def __init__(self, config: Config):
        self.config = config
        self.step_count = 0
        self.env = None

        yaml_path = self._YAML_CONFIGS.get(config.env_name)
        if yaml_path is None:
            raise ValueError(f"Unknown env_name '{config.env_name}'. "
                             f"Choose from: {list(self._YAML_CONFIGS)}")

        try:
            with open(yaml_path) as f:
                bb_config = yaml.safe_load(f)
            self.env = bb_utils.make_env(bb_config, training=True)
            obs, _ = self.env.reset()
            self._extract_dims(obs)
            # Override action_dim from the actual action space (not the Config default)
            config.action_dim = self.env.action_space.shape[0]
            print("[ENV] BabyBench loaded successfully.")
        except Exception as e:
            print(f"[ENV] BabyBench failed to load ({e}) — using mock environment.")

    def _extract_dims(self, obs):
        """Extract observation dimensions from actual BabyBench output."""
        if isinstance(obs, dict):
            # BabyBench returns 'observation' (proprioception) and 'touch'
            self.config.proprio_dim = obs.get(
                "observation", np.zeros(70)
            ).shape[0]
            self.config.touch_dim = obs.get(
                "touch", np.zeros(100)
            ).shape[0]

    def _flatten_obs(self, obs) -> np.ndarray:
        """Flatten observation dict into a single vector."""
        if isinstance(obs, dict):
            parts = []
            if "observation" in obs:          # proprioception key in BabyBench
                parts.append(obs["observation"])
            if "touch" in obs:
                parts.append(obs["touch"])
            if "eye_left" in obs and self.config.vision_dim > 0:
                parts.append(obs["eye_left"].flatten())
            return np.concatenate(parts).astype(np.float32)
        return np.asarray(obs, dtype=np.float32)

    def _compute_touch_reward(self, obs, info: dict) -> tuple:
        """
        Compute extrinsic reward from touch sensor data.

        === PROJECT NOTE ===
        The extrinsic reward is the "nurture" signal. We reward the agent
        for any self-touch contact. Specifically:
          - Each active touch sensor contributes to the reward
          - We also track WHICH body parts were touched for diversity analysis

        === IMPORTANT ===
        BabyBench's official goal is to learn self-touch WITHOUT extrinsic
        reward (purely intrinsic). Our extrinsic reward is an ADDITION
        for studying the nature/nurture balance. When α=1.0, this reward
        is not used at all (matching BabyBench's intended setup).
        """
        touch_data = info.get("touch_sensors", np.zeros(self.config.touch_dim))
        if isinstance(obs, dict) and "touch" in obs:
            touch_data = obs["touch"]

        # Count active touch sensors above threshold
        active_touches = (
            np.abs(touch_data) > self.config.touch_threshold
        )
        touch_count = active_touches.sum()

        # Reward is proportional to number of active touch sensors
        # Normalised by total sensors to keep reward in [0, 1]
        reward = touch_count / max(1, len(touch_data))

        # Track which body part groups were touched
        # (You'll need to map sensor indices to body parts for MIMo)
        parts_touched = set()
        n_sensors = len(touch_data)
        n_parts = len(self.config.body_part_groups)
        sensors_per_part = max(1, n_sensors // n_parts)
        for i, part in enumerate(self.config.body_part_groups):
            start = i * sensors_per_part
            end = min(start + sensors_per_part, n_sensors)
            if np.any(active_touches[start:end]):
                parts_touched.add(part)

        return reward, touch_count, parts_touched

    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""
        self.step_count = 0
        if self.env is not None:
            obs, _ = self.env.reset()          # Gymnasium API returns (obs, info)
            return self._flatten_obs(obs)
        else:
            # Mock environment: random initial state
            return np.random.randn(self.config.obs_dim).astype(np.float32) * 0.1

    def step(self, action: np.ndarray) -> tuple:
        """
        Take a step in the environment.

        Returns:
            obs: Next observation [obs_dim]
            reward_ext: Extrinsic (touch) reward
            done: Whether the episode is over
            info: Dictionary with touch details
        """
        self.step_count += 1

        if self.env is not None:
            # Gymnasium API: (obs, reward, terminated, truncated, info)
            obs, _, terminated, truncated, env_info = self.env.step(action)
            obs_flat = self._flatten_obs(obs)
            reward_ext, touch_count, parts_touched = (
                self._compute_touch_reward(obs, env_info)
            )
            done = terminated or truncated
        else:
            # Mock environment for testing
            obs_flat = np.random.randn(
                self.config.obs_dim
            ).astype(np.float32) * 0.1
            # Mock touch: random chance of touch proportional to action magnitude
            touch_prob = np.clip(np.abs(action).mean() * 0.3, 0, 0.5)
            touch_count = int(np.random.random() < touch_prob)
            reward_ext = float(touch_count) * 0.1
            parts_touched = set()
            if touch_count > 0:
                parts_touched.add(
                    np.random.choice(self.config.body_part_groups)
                )
            done = self.step_count >= self.config.max_episode_steps

        info = {
            "touch_count": touch_count,
            "parts_touched": parts_touched,
            "step": self.step_count,
        }

        return obs_flat, reward_ext, done, info


# ═══════════════════════════════════════════════════════════════════
# EXPERIENCE BUFFER
# ═══════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Stores one episode of experience for PPO updates.

    === WHY NOT A REPLAY BUFFER? ===
    PPO is an on-policy algorithm — it updates using experience collected
    under the CURRENT policy. Unlike DQN or SAC (off-policy), PPO can't
    reuse old experience because the importance weights would be too large.

    DreamerV3 IS off-policy for the world model (it uses a replay buffer),
    but its actor-critic is trained on imagined trajectories from the
    CURRENT world model, which is effectively on-policy.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards_ext = []
        self.rewards_int = []
        self.dones = []

    def add(self, obs, action, log_prob, reward_ext, reward_int, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards_ext.append(reward_ext)
        self.rewards_int.append(reward_int)
        self.dones.append(float(done))

    def get(self, next_obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Return all data as numpy arrays, ready for PPO update."""
        return {
            "obs": np.array(self.obs, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "rewards_ext": np.array(self.rewards_ext, dtype=np.float32),
            "rewards_int": np.array(self.rewards_int, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
            "next_obs": next_obs,
        }


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train(config: Config):
    """
    Main training function.

    === THE TRAINING LOOP STEP BY STEP ===

    For each episode:
      1. RESET the environment → get initial observation
      2. For each timestep:
         a. ENCODE observation through world model → latent z_t
         b. SELECT action via policy π(a|z_t)
         c. STEP environment → get next obs, touch info
         d. COMPUTE intrinsic reward via RND on the observation
         e. COMPUTE extrinsic reward from touch sensors
         f. STORE experience in rollout buffer
      3. After episode ends:
         a. TRAIN world model on the episode's experience
         b. UPDATE RND predictor
         c. COMBINE rewards using α
         d. UPDATE policy via PPO with dual value heads
         e. LOG all metrics

    This loop produces the data you need for your results section.
    """
    # ── Setup ──
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")
    print(f"[TRAIN] Alpha (nature/nurture balance): {config.alpha}")
    print(f"[TRAIN] Seed: {config.seed}")

    # Create results directory
    run_name = f"alpha_{config.alpha}_seed_{config.seed}"
    run_dir = os.path.join(config.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # ── Initialise components ──
    env = BabyBenchWrapper(config)

    world_model = WorldModel(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.world_model_hidden_dim,
        num_layers=config.world_model_layers,
    ).to(device)
    wm_optim = torch.optim.Adam(
        world_model.parameters(), lr=config.world_model_lr
    )

    rnd = RND(
        input_dim=config.obs_dim,
        hidden_dim=config.rnd_hidden_dim,
        output_dim=config.rnd_output_dim,
        num_layers=config.rnd_layers,
        learning_rate=config.rnd_lr,
    ).to(device)

    agent = PPOAgent(config)
    buffer = RolloutBuffer()

    # ── Metrics tracking ──
    metrics_log = []
    all_body_parts_ever_touched = set()

    print(f"\n{'='*60}")
    print(f"  TRAINING START — {config.num_episodes} episodes")
    print(f"  α = {config.alpha} "
          f"({'pure emergent' if config.alpha == 1.0 else 'pure cognitivist' if config.alpha == 0.0 else 'balanced'})")
    print(f"{'='*60}\n")

    # ══════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ══════════════════════════════════════════════════
    for episode in range(1, config.num_episodes + 1):
        obs = env.reset()
        buffer.clear()

        episode_reward_ext = 0
        episode_reward_int = 0
        episode_touches = 0
        episode_parts = set()
        episode_part_touches = {part: 0 for part in config.body_part_groups}
        episode_wm_loss = 0
        episode_rnd_loss = 0

        # ── Collect one episode of experience ──
        for step in range(config.max_episode_steps):

            # a. Select action
            action, log_prob = agent.select_action(obs)

            # b. Step environment
            next_obs, reward_ext, done, info = env.step(action)

            # c. Compute intrinsic reward via RND
            obs_normalised = rnd.normalise_obs(obs.reshape(1, -1))
            obs_t = obs_normalised.to(device)
            intrinsic_reward = rnd.compute_intrinsic_reward(obs_t)
            reward_int = intrinsic_reward.item()

            # Normalise intrinsic reward
            if config.normalise_rewards:
                reward_int_normalised = rnd.normalise_reward(
                    np.array([[reward_int]])
                )[0, 0]
            else:
                reward_int_normalised = reward_int

            # d. Store in buffer
            buffer.add(obs, action, log_prob,
                       reward_ext, reward_int_normalised, done)

            # e. Track metrics
            episode_reward_ext += reward_ext
            episode_reward_int += reward_int
            episode_touches += info["touch_count"]
            episode_parts.update(info["parts_touched"])
            for part in info["parts_touched"]:
                episode_part_touches[part] += 1

            obs = next_obs
            if done:
                break

        # ── Train world model on this episode's data ──
        rollout = buffer.get(next_obs=obs)
        obs_batch = torch.FloatTensor(rollout["obs"]).to(device)
        actions_batch = torch.FloatTensor(rollout["actions"]).to(device)
        rewards_batch = torch.FloatTensor(
            rollout["rewards_ext"]
        ).unsqueeze(-1).to(device)

        if len(rollout["obs"]) > 1:
            next_obs_batch = torch.FloatTensor(
                rollout["obs"][1:]
            ).to(device)

            wm_losses = world_model.compute_loss(
                obs_batch[:-1], actions_batch[:-1],
                next_obs_batch, rewards_batch[:-1]
            )
            wm_optim.zero_grad()
            wm_losses["total"].backward()
            nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
            wm_optim.step()
            episode_wm_loss = wm_losses["total"].item()

        # ── Update RND predictor ──
        obs_for_rnd = rnd.normalise_obs(rollout["obs"])
        obs_for_rnd_t = obs_for_rnd.to(device)
        episode_rnd_loss = rnd.update(obs_for_rnd_t)

        # ── Update policy via PPO ──
        ppo_metrics = agent.update(rollout)

        # ── Track body part coverage ──
        all_body_parts_ever_touched.update(episode_parts)

        # ── Adaptive α (stretch goal) ──
        if config.adaptive_alpha:
            # When RND error is high → lean on intrinsic motivation
            # When RND error is low → lean on extrinsic reward
            mean_rnd = episode_reward_int / max(1, step + 1)
            # Sigmoid to map RND error to [0.1, 0.9] range
            config.alpha = 0.1 + 0.8 / (1 + np.exp(-mean_rnd + 1))

        # ── Log metrics ──
        metrics = {
            "episode": episode,
            "reward_extrinsic": episode_reward_ext,
            "reward_intrinsic": episode_reward_int,
            "touch_count": episode_touches,
            "body_parts_this_ep": len(episode_parts),
            "body_parts_cumulative": len(all_body_parts_ever_touched),
            "body_part_touches": episode_part_touches,
            "world_model_loss": episode_wm_loss,
            "rnd_loss": episode_rnd_loss,
            "alpha": config.alpha,
            **ppo_metrics,
        }
        metrics_log.append(metrics)

        # ── Print progress ──
        if episode % config.log_interval == 0:
            print(
                f"[Ep {episode:4d}] "
                f"r_ext={episode_reward_ext:.3f} "
                f"r_int={episode_reward_int:.2f} "
                f"touches={episode_touches:3d} "
                f"parts={len(episode_parts)}/{len(config.body_part_groups)} "
                f"wm_loss={episode_wm_loss:.4f} "
                f"α={config.alpha:.2f}"
            )

        # ── Save checkpoint ──
        if episode % config.save_interval == 0:
            checkpoint = {
                "episode": episode,
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "world_model": world_model.state_dict(),
                "rnd_predictor": rnd.predictor.state_dict(),
                "config": vars(config),
            }
            path = os.path.join(run_dir, f"checkpoint_ep{episode}.pt")
            torch.save(checkpoint, path)
            print(f"  → Saved checkpoint: {path}")

    # ── Save final metrics ──
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2, default=str)
    print(f"\n[DONE] Metrics saved to {metrics_path}")
    print(f"[DONE] Total body parts discovered: "
          f"{len(all_body_parts_ever_touched)}/{len(config.body_part_groups)}")

    return metrics_log


# Need nn import for clip_grad_norm_ in the training loop
from torch import nn


# ═══════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train self-touch agent with world model + RND"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Nature/nurture balance (0=pure extrinsic, 1=pure intrinsic)"
    )
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--adaptive_alpha", action="store_true",
        help="Enable adaptive α based on RND signal"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory for saving results"
    )
    args = parser.parse_args()

    config = get_config(
        alpha=args.alpha,
        num_episodes=args.episodes,
        seed=args.seed,
        adaptive_alpha=args.adaptive_alpha,
        results_dir=args.results_dir,
    )

    train(config)


if __name__ == "__main__":
    main()
