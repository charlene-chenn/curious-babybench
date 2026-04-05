"""
rnd.py — Random Network Distillation for intrinsic motivation.

=== WHAT THIS IS ===
A module that generates curiosity-driven intrinsic rewards. The agent gets
rewarded for visiting states it hasn't seen before.

=== FRONTIER CONTEXT ===
RND (Burda et al., 2018) was a breakthrough for exploration in sparse-reward
environments. Before RND, the state of the art on Montezuma's Revenge (a
notoriously hard Atari game) was ~2500 points. RND achieved ~8000+, the first
method to surpass average human performance without demonstrations.

The core mechanism:
  1. FIXED target network f(s) — randomly initialised, NEVER trained
  2. TRAINABLE predictor network f̂(s) — trained to match the target
  3. Intrinsic reward = ||f̂(s) - f(s)||² (prediction error)

WHY IT WORKS:
  - States visited often → predictor has trained on them → low error → low reward
  - Novel states → predictor hasn't seen them → high error → high reward
  → Agent seeks out novel states!

=== PROJECT CONTEXT ===
RND is your "nature" / "emergent" signal. It implements the idea that infants
have an innate drive to explore and understand their world — not because
anyone tells them to, but because novelty itself is rewarding.

In your self-touch experiment:
  - Early in training: everything is novel → high RND reward everywhere
  - As the agent explores: familiar states become boring → agent seeks
    new sensations, including touching different body parts
  - The RND reward naturally decays as the agent builds its world model
  → This is the "self-improvement loop": curiosity drives exploration,
    exploration improves the world model, improved model reduces curiosity
    about known states, pushing the agent toward the unexplored.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class RunningMeanStd:
    """
    Running mean and standard deviation tracker.

    === WHY THIS MATTERS ===
    RND is extremely sensitive to observation scale. If proprioception values
    range from -3 to 3 but touch values range from 0 to 0.001, the predictor
    will focus entirely on proprioception (because the error is bigger there)
    and ignore touch — exactly the opposite of what we want for self-touch!

    This normaliser ensures all inputs have zero mean and unit variance,
    so the RND network treats all sensory modalities equally.

    The original RND paper (Burda et al., 2018) uses exactly this approach.
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Small initial count to avoid division by zero

    def update(self, batch: np.ndarray):
        """Update running statistics with a new batch of data."""
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        # Welford's online algorithm for numerically stable updates
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        # Parallel variance combination formula
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Normalise input using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class RNDNetwork(nn.Module):
    """
    A single MLP used for both the target and predictor networks.

    === ARCHITECTURAL CHOICE ===
    The original RND paper uses CNNs for Atari (image observations).
    For MIMo's vector observations (proprioception + touch), MLPs are
    more appropriate. We use the same architecture for both target and
    predictor — the only difference is that the target is FROZEN and
    the predictor is TRAINED.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                # Note: we use ReLU here (not SiLU) to match the original
                # RND paper. The target network's random features are more
                # diverse with ReLU activations.
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RND(nn.Module):
    """
    Random Network Distillation module.

    Produces intrinsic rewards based on state novelty.

    === HOW TO READ THE INTRINSIC REWARD ===
    - High reward (> 1.0): "I've never seen anything like this before!"
      The agent is in truly novel territory. In self-touch terms: the agent
      has discovered a new touch sensation or body configuration.

    - Medium reward (0.1 - 1.0): "This is somewhat familiar but not boring."
      The agent is in a partly explored region. It might be touching a body
      part it's only touched a few times before.

    - Low reward (< 0.1): "I know this state well."
      The agent has visited this state many times. The touch sensation is
      familiar. The agent should move on to explore something new.

    === FRONTIER NOTE: LIMITATIONS OF RND ===
    RND has a known failure mode called the "noisy TV problem": if part of
    the environment is stochastic (random), the predictor can NEVER learn to
    predict those states, so they remain perpetually "novel." The agent gets
    trapped watching noise.

    For MIMo/BabyBench, this is mostly not an issue because the physics are
    deterministic. But if you added visual observations, camera noise could
    trigger this. PreND (Davoodabadi et al., 2024) addresses this by using
    pre-trained representations that filter out noise.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 output_dim: int = 64, num_layers: int = 2,
                 learning_rate: float = 1e-4):
        super().__init__()

        # ── Target network: FIXED, randomly initialised ──
        # This is the "ground truth" that the predictor tries to match.
        # Because it's random, every input maps to a unique, unpredictable
        # output. The ONLY way to predict this output is to have seen
        # the input before (during training of the predictor).
        self.target = RNDNetwork(input_dim, hidden_dim, output_dim, num_layers)
        # Freeze target — it NEVER updates
        for param in self.target.parameters():
            param.requires_grad = False

        # ── Predictor network: TRAINABLE ──
        # Trained to match the target's output. For states it's trained on
        # (visited states), it matches well → low error. For novel states,
        # it can't predict → high error → high intrinsic reward.
        self.predictor = RNDNetwork(
            input_dim, hidden_dim, output_dim, num_layers
        )

        # Optimiser for the predictor only
        self.optimiser = torch.optim.Adam(
            self.predictor.parameters(), lr=learning_rate
        )

        # ── Observation normaliser ──
        # Critical for RND to work properly — see RunningMeanStd docs above.
        self.obs_normaliser = RunningMeanStd(shape=(input_dim,))

        # ── Reward normaliser ──
        # Keeps intrinsic rewards on a stable scale. Without this, intrinsic
        # rewards start enormous (everything is novel) and shrink over time,
        # making the α balance shift uncontrollably.
        self.reward_normaliser = RunningMeanStd(shape=(1,))

    def compute_intrinsic_reward(self, obs: torch.Tensor,
                                 normalise: bool = True) -> torch.Tensor:
        """
        Compute the intrinsic reward for a batch of observations.

        Args:
            obs: Observations [batch, obs_dim] — can be raw or latent
            normalise: Whether to normalise the reward

        Returns:
            intrinsic_reward: [batch, 1] — higher = more novel

        === THE CORE OF RND ===
        reward_i(s) = ||f̂(s) - f(s)||²

        That's it. The elegance of RND is its simplicity.
        """
        with torch.no_grad():
            target_features = self.target(obs)
        predictor_features = self.predictor(obs)

        # Prediction error = intrinsic reward
        # Squared L2 distance, summed over feature dimensions, per sample
        intrinsic_reward = (target_features - predictor_features).pow(2).sum(
            dim=-1, keepdim=True
        )

        return intrinsic_reward

    def update(self, obs: torch.Tensor) -> float:
        """
        Train the predictor network on a batch of observations.

        This is called AFTER collecting experience. The predictor learns
        to match the target for states the agent has visited, which means
        those states will produce lower intrinsic reward next time.

        === IMPORTANT SUBTLETY ===
        We only update the predictor on a FRACTION of the batch (25%).
        This is from the original RND paper — it prevents the predictor
        from learning too quickly, which would kill exploration too early.
        When scaling to more parallel environments (128+), they use even
        lower fractions (12.5%, 3.125%).

        Args:
            obs: Observations to train on [batch, obs_dim]

        Returns:
            prediction_loss: Scalar loss value for logging
        """
        # Subsample the batch (25% of observations)
        n = obs.shape[0]
        n_train = max(1, n // 4)
        indices = torch.randperm(n)[:n_train]
        obs_subset = obs[indices]

        with torch.no_grad():
            target_features = self.target(obs_subset)
        predictor_features = self.predictor(obs_subset)

        loss = F.mse_loss(predictor_features, target_features)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def normalise_obs(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        Normalise observations using running statistics and convert to tensor.

        Call this BEFORE compute_intrinsic_reward or update.
        Also updates the running statistics.
        """
        self.obs_normaliser.update(obs_np)
        normalised = self.obs_normaliser.normalise(obs_np)
        # Clip to prevent extreme values from corrupting the networks
        normalised = np.clip(normalised, -5.0, 5.0)
        return torch.FloatTensor(normalised)

    def normalise_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalise intrinsic rewards using running statistics.

        This keeps the intrinsic reward signal on a stable scale across
        training, making the α parameter meaningful at all stages.
        """
        self.reward_normaliser.update(reward)
        return reward / np.sqrt(self.reward_normaliser.var + 1e-8)
