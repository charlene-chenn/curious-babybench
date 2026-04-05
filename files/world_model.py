"""
world_model.py — Latent world model for self-touch learning.

=== WHAT THIS IS ===
A neural network that learns a compressed internal representation of the
agent's body and can predict what happens next. This is the "world model"
in your project title.

=== FRONTIER CONTEXT ===
This is a simplified version of the RSSM (Recurrent State-Space Model) from
DreamerV3 (Hafner et al., 2025). The full RSSM architecture:

  Observation o_t ──→ [Encoder] ──→ Posterior z_t (what IS happening)
                                        ↓
  Previous z_{t-1} + a_{t-1} ──→ [Dynamics] ──→ Prior ẑ_t (what SHOULD happen)
                                        ↓
                               [Decoder] ──→ Reconstructed ô_t
                               [Reward]  ──→ Predicted r̂_t

The encoder compresses high-dimensional observations into a compact latent
state z_t. The dynamics model predicts the next latent state from the current
one plus the action taken — this is the "imagination" capability that allows
DreamerV3 to train policies purely in latent space.

Our simplification: we remove the recurrent component (GRU) and the
stochastic/deterministic split, keeping the core encoder→dynamics→decoder
pipeline. This still gives us:
  - A compressed representation of body state
  - A forward model for predicting consequences of actions
  - Reconstruction quality as a metric for "how well does the agent
    understand its own body?"

=== PROJECT CONTEXT ===
The world model quality is one of your key dependent variables. Your hypothesis
is that agents with balanced α (mixing curiosity and reward) will develop
BETTER world models because they explore more diverse states, giving the
encoder/decoder more varied training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int,
              num_layers: int = 2, activation: str = "silu") -> nn.Sequential:
    """
    Build a multi-layer perceptron.

    We use SiLU (Sigmoid Linear Unit) as the activation — this is what
    DreamerV3 uses instead of ReLU. SiLU is smoother, which helps with
    gradient flow in deeper networks.

    === FRONTIER NOTE ===
    DreamerV3 also uses LayerNorm after each hidden layer for training
    stability. We include it here following their design.
    """
    act_fn = nn.SiLU if activation == "silu" else nn.ReLU
    layers = []
    dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # No activation/norm on output layer
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act_fn())
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    """
    Latent world model with encoder, dynamics predictor, decoder, and
    reward predictor.

    The model learns to:
    1. ENCODE observations into a compact latent space (z_t)
    2. PREDICT what happens next given an action (z_{t+1} = f(z_t, a_t))
    3. DECODE latent states back to observations (for training signal)
    4. PREDICT reward from latent states

    === HOW THIS CONNECTS TO YOUR NARRATIVE ===
    - The encoder learns "what is my body doing right now?"
    - The dynamics model learns "if I move this joint, what happens?"
    - The decoder accuracy tells us "how well does the agent understand itself?"
    - The reward predictor learns "which states lead to touch?"
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # ── Encoder: observation → latent state ──
        # Maps the raw sensor readings (proprio + touch + optional vision)
        # into a compact vector z_t. This is lossy compression — the model
        # must learn WHAT to keep and what to discard.
        self.encoder = build_mlp(obs_dim, hidden_dim, latent_dim, num_layers)

        # ── Dynamics predictor: (z_t, a_t) → z_{t+1} ──
        # The "imagination engine." Given current latent state and an action,
        # predict the next latent state. In DreamerV3, the actor-critic trains
        # entirely on rollouts from this model.
        #
        # === FRONTIER NOTE ===
        # DreamerV3 uses a GRU for this, maintaining a recurrent hidden state
        # across timesteps. This lets it capture long-term dependencies.
        # Our version is a feedforward model (one-step prediction only),
        # which is simpler but can still capture immediate dynamics.
        self.dynamics = build_mlp(
            latent_dim + action_dim, hidden_dim, latent_dim, num_layers
        )

        # ── Decoder: z_t → reconstructed observation ──
        # Reconstructs the observation from the latent state.
        # The reconstruction error is the main training signal for the
        # encoder — it forces the encoder to preserve useful information.
        #
        # === PROJECT NOTE ===
        # We decode proprio and touch separately, because they have
        # different scales and semantics. This also lets you measure
        # "does the agent understand its joint positions?" vs
        # "does the agent understand touch sensations?" separately.
        self.decoder_proprio = build_mlp(
            latent_dim, hidden_dim, obs_dim, num_layers
        )

        # ── Reward predictor: z_t → predicted reward ──
        # Learns which latent states are associated with reward (touch).
        # This is used for the world model's training, not for policy learning.
        self.reward_predictor = build_mlp(
            latent_dim, hidden_dim, 1, num_layers
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode an observation into the latent space.

        Args:
            obs: Raw observation [batch, obs_dim]

        Returns:
            z: Latent state [batch, latent_dim]

        This is the agent's "perception" — compressing the rich sensory
        input into a manageable internal representation.
        """
        return self.encoder(obs)

    def predict_next(self, z: torch.Tensor,
                     action: torch.Tensor) -> torch.Tensor:
        """
        Predict the next latent state given current state and action.

        Args:
            z: Current latent state [batch, latent_dim]
            action: Action taken [batch, action_dim]

        Returns:
            z_next: Predicted next latent state [batch, latent_dim]

        === FRONTIER NOTE ===
        In DreamerV3, this is where "dreaming" happens. The agent can
        chain these predictions: z_0 → z_1 → z_2 → ... → z_H, imagining
        entire trajectories without interacting with the real environment.
        This is why model-based RL is so sample-efficient — you get
        thousands of imagined experiences from a single real one.
        """
        return self.dynamics(torch.cat([z, action], dim=-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct observation from latent state.

        Args:
            z: Latent state [batch, latent_dim]

        Returns:
            obs_recon: Reconstructed observation [batch, obs_dim]
        """
        return self.decoder_proprio(z)

    def predict_reward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict reward from latent state.

        Args:
            z: Latent state [batch, latent_dim]

        Returns:
            reward: Predicted reward [batch, 1]
        """
        return self.reward_predictor(z)

    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor,
                     next_obs: torch.Tensor,
                     rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all world model losses on a batch of real experience.

        This trains the world model to:
        1. Accurately reconstruct observations (reconstruction loss)
        2. Accurately predict next states (dynamics loss)
        3. Accurately predict rewards (reward loss)

        === PROJECT NOTE ===
        The TOTAL loss is what you report as "world model quality" in your
        results. Lower loss = the agent has a better internal model of its
        body. Your hypothesis: balanced α → more diverse experience →
        lower world model loss.

        Args:
            obs: Current observations [batch, obs_dim]
            actions: Actions taken [batch, action_dim]
            next_obs: Next observations [batch, obs_dim]
            rewards: Rewards received [batch, 1]

        Returns:
            Dictionary of individual and total losses
        """
        # Encode current and next observations
        z = self.encode(obs)
        z_next_true = self.encode(next_obs)

        # 1. Reconstruction loss — can the decoder recover the observation?
        obs_recon = self.decode(z)
        recon_loss = F.mse_loss(obs_recon, obs)

        # 2. Dynamics loss — can the dynamics model predict the next state?
        z_next_pred = self.predict_next(z, actions)
        dynamics_loss = F.mse_loss(z_next_pred, z_next_true.detach())
        # Note: .detach() on z_next_true means we don't backpropagate the
        # dynamics loss through the encoder of the next observation.
        # This is a common choice that keeps training stable.

        # 3. Reward prediction loss
        reward_pred = self.predict_reward(z)
        reward_loss = F.mse_loss(reward_pred, rewards)

        # === FRONTIER NOTE ===
        # DreamerV3 also includes a KL divergence term that regularises
        # the latent space (similar to a VAE). They use "KL balancing"
        # where the KL loss is split 80/20 between posterior and prior,
        # preventing the common problem of posterior collapse. We omit
        # this for simplicity since our latent space is deterministic.

        total_loss = recon_loss + dynamics_loss + 0.5 * reward_loss

        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "dynamics": dynamics_loss,
            "reward": reward_loss,
        }

    def get_latent(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the latent representation without gradients (for RND input).

        This is used to feed the world model's compressed state into the
        RND module — the agent's curiosity operates on its INTERNAL
        representation, not raw observations. This is more biologically
        plausible: an infant's curiosity is driven by what it understands,
        not raw sensory data.
        """
        with torch.no_grad():
            return self.encode(obs)
