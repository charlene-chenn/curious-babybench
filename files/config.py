"""
config.py — Centralised hyperparameters for the self-touch world model project.

=== FRONTIER CONTEXT ===
DreamerV3 (Hafner et al., 2025) showed that a SINGLE set of hyperparameters
can work across 150+ diverse tasks. Their secret: normalisation tricks that
make the algorithm robust to reward scale, observation scale, and action space
differences. We adopt that philosophy here with:
  - Observation normalisation (running mean/std)
  - Percentile-based return normalisation in PPO
  - Symlog transform for large value targets

=== PROJECT CONTEXT ===
The key parameter is `alpha` — your independent variable that controls the
nature (intrinsic/emergent) vs nurture (extrinsic/cognitivist) balance.
Everything else should stay fixed across your experimental conditions.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ──────────────────────────────────────────────
    # EXPERIMENT CONTROL (your independent variable)
    # ──────────────────────────────────────────────
    alpha: float = 0.5
    """
    The nature/nurture dial.
      α = 1.0  → pure emergent (only RND curiosity, no touch reward)
      α = 0.75 → mostly curiosity-driven
      α = 0.5  → equal balance
      α = 0.25 → mostly reward-guided
      α = 0.0  → pure cognitivist (only touch reward, no curiosity)

    This is the ONLY thing you change between experimental conditions.
    """

    adaptive_alpha: bool = False
    """
    Stretch goal: if True, α is dynamically adjusted based on RND prediction
    error. High error → more intrinsic motivation. Low error → more extrinsic.
    This implements the 'self-improvement loop' discussed in the walkthrough.
    """

    seed: int = 42
    """Random seed for reproducibility. Run each α with seeds [1, 2, 3]."""

    # ──────────────────────────────────────────────
    # ENVIRONMENT (BabyBench / MIMo)
    # ──────────────────────────────────────────────
    env_name: str = "self_touch"
    """BabyBench environment variant. Options: 'self_touch', 'hand_regard'."""

    max_episode_steps: int = 1000
    """Steps per episode. MIMo simulates at ~50Hz, so 1000 steps ≈ 20 seconds."""

    num_episodes: int = 500
    """Total training episodes. Increase for better results if compute allows."""

    # ──────────────────────────────────────────────
    # OBSERVATION / ACTION SPACES
    # ──────────────────────────────────────────────
    # These will be overwritten at runtime from the actual BabyBench env,
    # but we set defaults for testing and documentation.

    proprio_dim: int = 70
    """Proprioception: joint angles + velocities for MIMo's ~35 joints."""

    touch_dim: int = 100
    """Touch sensors distributed across MIMo's body surface."""

    vision_dim: int = 0
    """
    Set to 0 to disable vision (faster training, simpler model).
    Self-touch can be learned from proprioception + touch alone.
    Set to e.g. 64*64*3 = 12288 if you want to include egocentric vision.
    """

    action_dim: int = 35
    """Joint torques — one per actuated joint in MIMo."""

    @property
    def obs_dim(self) -> int:
        """Total observation dimensionality (proprio + touch + vision)."""
        return self.proprio_dim + self.touch_dim + self.vision_dim

    # ──────────────────────────────────────────────
    # WORLD MODEL (RSSM-inspired)
    # ──────────────────────────────────────────────
    # === FRONTIER NOTE ===
    # DreamerV3 uses a full RSSM with:
    #   - Deterministic path: GRU hidden state (recurrence)
    #   - Stochastic path: categorical latent variables (32 classes × 32 dims)
    #   - KL balancing to prevent posterior collapse
    # Our simplified version uses a deterministic encoder-decoder with a
    # transition model, which captures the core idea without the complexity.

    latent_dim: int = 128
    """
    Size of the compressed world state z_t.
    Think of this as "how much the agent can remember about the current moment."
    DreamerV3 uses 32×32=1024 categorical dims; we use a simpler continuous vector.
    """

    world_model_hidden_dim: int = 256
    """Hidden layer size in the world model networks."""

    world_model_lr: float = 3e-4
    """Learning rate for the world model (encoder + dynamics + decoder)."""

    world_model_layers: int = 2
    """Number of hidden layers in each world model sub-network."""

    # ──────────────────────────────────────────────
    # RND (Random Network Distillation)
    # ──────────────────────────────────────────────
    # === FRONTIER NOTE ===
    # The original RND paper (Burda et al., 2018) used a CNN for Atari.
    # For MIMo's sensor data (vectors, not images), we use MLPs.
    # PreND (Davoodabadi et al., 2024) showed that using pre-trained
    # representations in the target network improves RND — a potential
    # extension for your project.

    rnd_output_dim: int = 64
    """
    Dimensionality of the RND embedding space.
    Both the fixed target and trainable predictor map observations to this space.
    The L2 distance between their outputs IS the intrinsic reward.
    """

    rnd_hidden_dim: int = 256
    """Hidden layer size in RND networks."""

    rnd_lr: float = 1e-4
    """
    Learning rate for the RND predictor network.
    Deliberately lower than the policy LR — we want the predictor to learn
    slowly so that novel states remain rewarding for longer.
    """

    rnd_layers: int = 2
    """Number of hidden layers in RND networks."""

    # ──────────────────────────────────────────────
    # PPO (Proximal Policy Optimisation)
    # ──────────────────────────────────────────────
    # === FRONTIER NOTE ===
    # PPO (Schulman et al., 2017) remains the most widely used policy gradient
    # algorithm. DreamerV3 uses it for the actor-critic trained in imagination.
    # The dual value heads come from the RND paper's insight that intrinsic
    # and extrinsic rewards have different temporal properties.

    policy_lr: float = 3e-4
    """Learning rate for the actor (policy network)."""

    value_lr: float = 1e-3
    """Learning rate for the critic (value networks)."""

    gamma_ext: float = 0.99
    """
    Discount factor for extrinsic rewards.
    Standard value — the agent cares about long-term touch outcomes.
    """

    gamma_int: float = 0.99
    """
    Discount factor for intrinsic rewards.
    The RND paper uses 0.99 for non-episodic intrinsic returns.
    Some implementations use a different γ (e.g., 0.999) to encourage
    even longer-horizon exploration.
    """

    gae_lambda: float = 0.95
    """GAE (Generalised Advantage Estimation) λ. Standard value."""

    clip_eps: float = 0.2
    """PPO clipping parameter. Standard value."""

    ppo_epochs: int = 4
    """Number of PPO update passes per batch of collected experience."""

    batch_size: int = 64
    """Minibatch size for PPO updates."""

    hidden_dim: int = 256
    """Hidden layer size in actor and critic networks."""

    policy_layers: int = 2
    """Number of hidden layers in the policy network."""

    value_layers: int = 2
    """Number of hidden layers in the value networks."""

    # ──────────────────────────────────────────────
    # NORMALISATION
    # ──────────────────────────────────────────────

    normalise_obs: bool = True
    """
    Apply running mean/std normalisation to observations before feeding to
    all networks. Critical for RND (from the original paper) and generally
    good practice for PPO.
    """

    normalise_rewards: bool = True
    """
    Apply running std normalisation to intrinsic rewards.
    Prevents the RND signal from dominating early in training when
    everything is novel and prediction errors are huge.
    """

    # ──────────────────────────────────────────────
    # LOGGING & EVALUATION
    # ──────────────────────────────────────────────
    log_interval: int = 10
    """Print metrics every N episodes."""

    eval_interval: int = 50
    """Run evaluation (no exploration noise) every N episodes."""

    save_interval: int = 100
    """Save model checkpoint every N episodes."""

    results_dir: str = "results"
    """Directory for saving logs, checkpoints, and figures."""

    render_eval: bool = False
    """Whether to render videos during evaluation episodes."""

    # ──────────────────────────────────────────────
    # TOUCH-SPECIFIC METRICS
    # ──────────────────────────────────────────────
    touch_threshold: float = 0.01
    """
    Minimum sensor activation to count as a "touch event."
    MIMo's touch sensors return continuous values — this threshold
    binarises them for counting touches.
    """

    body_part_groups: List[str] = field(default_factory=lambda: [
        "head", "torso", "left_arm", "right_arm",
        "left_hand", "right_hand", "left_leg", "right_leg"
    ])
    """
    Named body regions for touch diversity analysis.
    You'll map MIMo's touch sensor indices to these groups.
    """


def get_config(**overrides) -> Config:
    """Create a config with optional overrides from command line or sweep."""
    cfg = Config()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return cfg
