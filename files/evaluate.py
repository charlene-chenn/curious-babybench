"""
evaluate.py — Generate analysis plots for the self-touch world model project.

=== WHAT THIS IS ===
After training with different α values, run this script to produce the
figures you need for your 3-minute video:
  1. Learning curves per α condition
  2. Touch count comparison across conditions
  3. Body part coverage diversity
  4. World model quality over time
  5. Summary comparison table

=== HOW TO RUN ===
    python evaluate.py --results_dir results/

This will scan all runs in results/ and produce comparative plots.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
from collections import defaultdict
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_MIMO_ROOT = os.path.join(_REPO_ROOT, "MIMo")
for _path in (_HERE, _REPO_ROOT, _MIMO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import mimoEnv
import babybench.utils as bb_utils
import babybench.eval as bb_eval
from agent import Actor

# Try to import matplotlib — if not available, we'll create data-only output
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found — will generate data summaries only.")
    print("       Install with: pip install matplotlib")


def load_all_runs(results_dir: str) -> Dict[float, List[dict]]:
    """
    Load metrics from all runs, grouped by α value.

    Returns:
        {alpha_value: [list of metric dicts per episode]}
    """
    runs_by_alpha = defaultdict(list)

    for run_name in os.listdir(results_dir):
        run_path = os.path.join(results_dir, run_name)
        metrics_path = os.path.join(run_path, "metrics.json")

        if not os.path.isfile(metrics_path):
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Extract alpha from config or run name
        config_path = os.path.join(run_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            alpha = config.get("alpha", 0.5)
        else:
            # Parse from directory name: "alpha_0.5_seed_42"
            try:
                alpha = float(run_name.split("_")[1])
            except (IndexError, ValueError):
                alpha = 0.5

        runs_by_alpha[alpha].append(metrics)

    return dict(sorted(runs_by_alpha.items()))


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average for smoother learning curves."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def extract_metric(runs: List[dict], key: str) -> np.ndarray:
    """
    Extract a metric across all runs, averaging over seeds.

    Returns: [episodes] array of mean values
    """
    all_values = []
    for run in runs:
        values = [ep.get(key, 0) for ep in run]
        all_values.append(values)

    # Pad to same length
    max_len = max(len(v) for v in all_values)
    padded = np.zeros((len(all_values), max_len))
    for i, v in enumerate(all_values):
        padded[i, :len(v)] = v

    return padded.mean(axis=0)


def plot_learning_curves(runs_by_alpha: dict, save_dir: str):
    """
    Plot 1: Learning curves for each α condition.

    === YOUR VIDEO NARRATIVE ===
    "Here we see the cumulative extrinsic reward (touch) over training
    for each α condition. The balanced conditions (α=0.25–0.75) learn
    faster because curiosity drives the agent to discover touch events
    earlier, while the task reward keeps it from wandering aimlessly."
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = {
        0.0: "#E24B4A",    # Red — pure cognitivist
        0.25: "#D85A30",   # Coral
        0.5: "#639922",    # Green — balanced
        0.75: "#1D9E75",   # Teal
        1.0: "#378ADD",    # Blue — pure emergent
    }

    # Panel 1: Extrinsic reward (touch)
    ax = axes[0]
    for alpha, runs in runs_by_alpha.items():
        values = smooth(extract_metric(runs, "reward_extrinsic"))
        color = colors.get(alpha, "#888780")
        ax.plot(values, label=f"α={alpha}", color=color, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Extrinsic reward (touch)")
    ax.set_title("Touch reward over training")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Intrinsic reward (RND)
    ax = axes[1]
    for alpha, runs in runs_by_alpha.items():
        values = smooth(extract_metric(runs, "reward_intrinsic"))
        color = colors.get(alpha, "#888780")
        ax.plot(values, label=f"α={alpha}", color=color, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Intrinsic reward (RND)")
    ax.set_title("Curiosity signal over training")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: World model loss
    ax = axes[2]
    for alpha, runs in runs_by_alpha.items():
        values = smooth(extract_metric(runs, "world_model_loss"))
        color = colors.get(alpha, "#888780")
        ax.plot(values, label=f"α={alpha}", color=color, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("World model loss")
    ax.set_title("World model quality")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_touch_analysis(runs_by_alpha: dict, save_dir: str):
    """
    Plot 2: Touch count and body part diversity comparison.

    === YOUR VIDEO NARRATIVE ===
    "The bar chart on the left shows total self-touch events per condition.
    Pure cognitivist (α=0) achieves many touches but they're repetitive —
    the agent finds one easy touch and exploits it. The bar chart on the
    right shows body part diversity: the curious agents (higher α) touch
    MORE DIFFERENT body parts, even if their total count is lower."
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    alphas = sorted(runs_by_alpha.keys())
    colors = ["#E24B4A", "#D85A30", "#639922", "#1D9E75", "#378ADD"]

    # Panel 1: Total touches per condition
    ax = axes[0]
    touch_totals = []
    for alpha in alphas:
        values = extract_metric(runs_by_alpha[alpha], "touch_count")
        touch_totals.append(values.sum())
    bars = ax.bar(
        [f"α={a}" for a in alphas], touch_totals,
        color=colors[:len(alphas)], edgecolor="white", linewidth=0.5
    )
    ax.set_ylabel("Total touch events")
    ax.set_title("Self-touch frequency")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Body part diversity
    ax = axes[1]
    diversity = []
    for alpha in alphas:
        values = extract_metric(
            runs_by_alpha[alpha], "body_parts_cumulative"
        )
        # Final cumulative value
        diversity.append(values[-1] if len(values) > 0 else 0)
    bars = ax.bar(
        [f"α={a}" for a in alphas], diversity,
        color=colors[:len(alphas)], edgecolor="white", linewidth=0.5
    )
    ax.set_ylabel("Body parts discovered")
    ax.set_title("Touch diversity")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "touch_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def extract_body_part_touches(runs: List[dict], body_parts: List[str]) -> Dict[str, float]:
    """
    Sum up per-body-part touch counts across all episodes and seeds.

    Returns: {body_part: total_touches} averaged over seeds
    """
    totals = {part: 0.0 for part in body_parts}
    n_seeds = len(runs)
    for run in runs:
        for ep in run:
            bp_touches = ep.get("body_part_touches", {})
            for part in body_parts:
                totals[part] += bp_touches.get(part, 0)
    if n_seeds > 0:
        totals = {part: count / n_seeds for part, count in totals.items()}
    return totals


def plot_body_part_touches(runs_by_alpha: dict, save_dir: str):
    """
    Plot per-body-part touch frequency for each α condition.

    Shows a grouped bar chart so you can compare which body parts each
    condition touches most, revealing whether curiosity-driven agents
    explore more diverse regions of the body.
    """
    if not HAS_MATPLOTLIB:
        return

    body_parts = [
        "head", "torso", "left_arm", "right_arm",
        "left_hand", "right_hand", "left_leg", "right_leg"
    ]
    # Try to detect body parts from data
    for alpha, runs in runs_by_alpha.items():
        for ep in runs[0]:
            bp = ep.get("body_part_touches", {})
            if bp:
                body_parts = list(bp.keys())
                break
        break

    alphas = sorted(runs_by_alpha.keys())
    colors = {
        0.0: "#E24B4A",
        0.25: "#D85A30",
        0.5: "#639922",
        0.75: "#1D9E75",
        1.0: "#378ADD",
    }

    n_parts = len(body_parts)
    n_alphas = len(alphas)
    bar_width = 0.8 / n_alphas
    x = np.arange(n_parts)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, alpha in enumerate(alphas):
        touches = extract_body_part_touches(runs_by_alpha[alpha], body_parts)
        counts = [touches[part] for part in body_parts]
        offset = (i - n_alphas / 2 + 0.5) * bar_width
        color = colors.get(alpha, "#888780")
        ax.bar(x + offset, counts, bar_width, label=f"α={alpha}",
               color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(body_parts, rotation=30, ha="right")
    ax.set_ylabel("Total touch count (avg over seeds)")
    ax.set_title("Body part touch frequency by condition")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "body_part_touches.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def print_summary_table(runs_by_alpha: dict):
    """
    Print a summary comparison table.

    === YOUR VIDEO NARRATIVE ===
    This table is perfect for showing side-by-side results in your video.
    It maps directly to the "Results, interpretations, and impact" section.
    """
    print("\n" + "=" * 78)
    print("  RESULTS SUMMARY")
    print("=" * 78)
    print(f"{'α':>6} | {'Condition':<20} | {'Touches':>8} | "
          f"{'Parts':>6} | {'WM Loss':>8} | {'Final r_ext':>10}")
    print("-" * 78)

    for alpha in sorted(runs_by_alpha.keys()):
        runs = runs_by_alpha[alpha]
        touches = extract_metric(runs, "touch_count").sum()
        parts = extract_metric(runs, "body_parts_cumulative")
        final_parts = parts[-1] if len(parts) > 0 else 0
        wm_loss = extract_metric(runs, "world_model_loss")
        final_wm = wm_loss[-1] if len(wm_loss) > 0 else 0
        r_ext = extract_metric(runs, "reward_extrinsic")
        final_r = r_ext[-1] if len(r_ext) > 0 else 0

        if alpha == 0.0:
            condition = "Pure cognitivist"
        elif alpha == 1.0:
            condition = "Pure emergent"
        else:
            condition = f"Balanced ({alpha})"

        print(f"{alpha:>6.2f} | {condition:<20} | {touches:>8.0f} | "
              f"{final_parts:>6.0f} | {final_wm:>8.4f} | {final_r:>10.4f}")

    print("=" * 78)
    print()


def generate_report(results_dir: str):
    """Generate all evaluation outputs."""
    print(f"\n[EVAL] Loading runs from {results_dir}/\n")

    runs_by_alpha = load_all_runs(results_dir)

    if not runs_by_alpha:
        print("[EVAL] No completed runs found. Run train.py first.")
        print("       Example: python train.py --alpha 0.5 --episodes 100")
        return

    print(f"[EVAL] Found {len(runs_by_alpha)} α conditions:")
    for alpha, runs in runs_by_alpha.items():
        n_seeds = len(runs)
        n_eps = len(runs[0]) if runs else 0
        print(f"  α={alpha}: {n_seeds} seed(s), {n_eps} episodes each")

    # Create output directory
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate all outputs
    print("\n[EVAL] Generating plots...")
    plot_learning_curves(runs_by_alpha, plots_dir)
    plot_touch_analysis(runs_by_alpha, plots_dir)
    plot_body_part_touches(runs_by_alpha, plots_dir)
    print_summary_table(runs_by_alpha)

    print(f"[EVAL] All plots saved to {plots_dir}/")
    print("[EVAL] Use these in your 3-minute video!")


def flatten_obs(obs):
    """Flatten observation dict into a single vector (matches BabyBenchWrapper)."""
    if isinstance(obs, dict):
        parts = []
        if "observation" in obs:
            parts.append(obs["observation"])
        if "touch" in obs:
            parts.append(obs["touch"])
        return np.concatenate(parts).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)


def load_policy(checkpoint_path, device):
    """Load a trained Actor from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    actor = Actor(
        obs_dim=cfg["proprio_dim"] + cfg["touch_dim"] + cfg["vision_dim"],
        action_dim=cfg["action_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["policy_layers"],
    ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    print(f"[EVAL] Loaded policy from {checkpoint_path}")
    print(f"[EVAL] Checkpoint was from episode {checkpoint['episode']}, "
          f"alpha={cfg['alpha']}, seed={cfg['seed']}")
    return actor


def select_action(actor, obs, device):
    """Select a deterministic action from the trained policy."""
    obs_flat = flatten_obs(obs)
    obs_t = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
    with torch.no_grad():
        action, _ = actor.get_action(obs_t, deterministic=True)
    return action.cpu().numpy().squeeze(0)


def record_video(config_path, checkpoint_path=None, episodes=3, duration=1000):
    """
    Run evaluation episodes with video recording.

    Args:
        config_path: Path to the BabyBench YAML config (e.g. config_selftouch.yml)
        checkpoint_path: Path to a trained .pt checkpoint. If None, uses random actions.
        episodes: Number of evaluation episodes to record.
        duration: Timesteps per episode.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained policy if checkpoint provided
    actor = None
    if checkpoint_path is not None:
        actor = load_policy(checkpoint_path, device)
    else:
        print("[EVAL] No checkpoint provided — using random actions.")

    env = bb_utils.make_env(config, training=False)
    env.reset()

    evaluation = bb_eval.EVALS[config['behavior']](
        env=env,
        duration=duration,
        render=True,
        save_dir=config['save_dir'],
    )

    evaluation.eval_logs()

    for ep_idx in range(episodes):
        print(f'Running evaluation episode {ep_idx+1}/{episodes}')

        obs, _ = env.reset()
        evaluation.reset()

        for t_idx in range(duration):
            if actor is not None:
                action = select_action(actor, obs, device)
            else:
                action = env.action_space.sample()

            obs, _, _, _, info = env.step(action)
            evaluation.eval_step(info)

        evaluation.end(episode=ep_idx)

    print(f"[EVAL] Videos saved to {config['save_dir']}/videos/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate self-touch experiment results"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory containing training runs"
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Record evaluation videos instead of generating plots"
    )
    parser.add_argument(
        "--config", type=str, default="config_selftouch.yml",
        help="BabyBench YAML config for video recording"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained checkpoint (.pt) for video recording"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of evaluation episodes for video recording"
    )
    parser.add_argument(
        "--duration", type=int, default=1000,
        help="Timesteps per evaluation episode"
    )
    args = parser.parse_args()

    if args.video:
        record_video(args.config, args.checkpoint, args.episodes, args.duration)
    else:
        generate_report(args.results_dir)
