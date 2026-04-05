"""
agent.py — PPO agent with dual value heads for intrinsic/extrinsic rewards.

=== WHAT THIS IS ===
The decision-making core of the agent. It has:
  - An ACTOR (policy) that decides which joint torques to apply
  - Two CRITICS (value heads) that estimate future reward:
      • V_e: estimates return from extrinsic reward (touch)
      • V_i: estimates return from intrinsic reward (RND novelty)
  - A REWARD COMBINER that mixes intrinsic and extrinsic using α

=== FRONTIER CONTEXT ===
PPO (Proximal Policy Optimisation, Schulman et al. 2017) is the most widely
used policy gradient algorithm in modern RL. It's used in:
  - DreamerV3's actor-critic (trained in imagination)
  - OpenAI's RLHF pipeline for training ChatGPT
  - Virtually all robotics RL papers since 2018

The dual value heads come from the RND paper (Burda et al., 2018). The key
insight: intrinsic and extrinsic rewards have DIFFERENT temporal properties.

  Extrinsic reward is EPISODIC:
    - When the episode ends, the return resets to zero
    - The agent learns "touching my arm gives me reward THIS episode"

  Intrinsic reward is NON-EPISODIC:
    - A state that's novel now was ALWAYS novel at that point
    - The return doesn't reset at episode boundaries
    - The agent learns "this whole region of state space is unexplored"

Using a single value function for both conflates these timescales and
leads to poor estimates. Separate heads let each learn at its own pace.

=== PROJECT CONTEXT ===
This is where nature meets nurture. The reward combiner implements:

    r_total = α × r_intrinsic + (1 - α) × r_extrinsic

By sweeping α, you test the cognitivist vs emergentist hypotheses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from config import Config


class Actor(nn.Module):
    """
    The policy network — maps observations to action distributions.

    === ACTION SPACE ===
    MIMo has continuous joint torques. We parameterise the policy as a
    Gaussian: the network outputs a mean μ and log-std σ for each joint,
    and we sample actions from N(μ, σ²).

    The log-std is clamped to prevent extremely large or small variances.
    Large variance → random flailing. Small variance → the agent stops
    exploring. The clamp range [-2, 2] keeps σ between ~0.14 and ~7.4.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()

        # Shared feature extractor
        layers = []
        dims = [obs_dim] + [hidden_dim] * num_layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.SiLU())
        self.backbone = nn.Sequential(*layers)

        # Mean head — deterministic part of the action
        self.mean_head = nn.Linear(hidden_dim, action_dim)

        # Log-std head — exploration noise
        # Initialised to produce σ ≈ 0.5 (moderate exploration)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.log_std_head.bias, -0.7)  # exp(-0.7) ≈ 0.5

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action distribution parameters.

        Returns:
            mean: Action mean [batch, action_dim]
            log_std: Action log-std [batch, action_dim]
        """
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -2.0, 2.0)
        return mean, log_std

    def get_action(self, obs: torch.Tensor,
                   deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute its log-probability.

        Args:
            obs: Observation [batch, obs_dim]
            deterministic: If True, return the mean (for evaluation)

        Returns:
            action: Sampled action [batch, action_dim]
            log_prob: Log-probability of the action [batch, 1]
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        if deterministic:
            action = mean
        else:
            # Reparameterisation trick: sample from N(0,1) and scale
            noise = torch.randn_like(mean)
            action = mean + std * noise

        # Log-probability of a Gaussian
        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-8)).pow(2)
            + 2 * log_std
            + np.log(2 * np.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Clip actions to valid range (MIMo joint torques)
        action = torch.clamp(action, -1.0, 1.0)

        return action, log_prob

    def evaluate_action(self, obs: torch.Tensor,
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the log-probability and entropy of a given action.
        Used during PPO updates (we need log_prob of OLD actions under NEW policy).

        Returns:
            log_prob: [batch, 1]
            entropy: [batch, 1] — higher entropy = more exploration
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-8)).pow(2)
            + 2 * log_std
            + np.log(2 * np.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Entropy of a Gaussian: 0.5 * log(2πe σ²) per dimension
        entropy = 0.5 * (1 + 2 * log_std + np.log(2 * np.pi))
        entropy = entropy.sum(dim=-1, keepdim=True)

        return log_prob, entropy


class DualCritic(nn.Module):
    """
    Dual value heads: V_e (extrinsic) and V_i (intrinsic).

    === WHY TWO HEADS? ===
    From the RND paper (Burda et al., 2018):

    "We use two value heads: one that estimates return from extrinsic
    (episodic) rewards, and one for intrinsic (non-episodic) rewards."

    The extrinsic value head uses γ_ext with episodic returns (reset at
    episode end). The intrinsic value head uses γ_int with non-episodic
    returns (no reset — novelty transcends episode boundaries).

    This separation is critical because:
    - V_e learns "which states lead to touch in this episode?"
    - V_i learns "which states are globally unexplored?"
    These are fundamentally different questions with different timescales.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2):
        super().__init__()

        # Shared feature extractor (optional — separate also works)
        layers = []
        dims = [obs_dim] + [hidden_dim] * num_layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.SiLU())
        self.backbone = nn.Sequential(*layers)

        # Separate value heads
        self.v_extrinsic = nn.Linear(hidden_dim, 1)
        self.v_intrinsic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate both value functions.

        Returns:
            v_ext: Extrinsic value estimate [batch, 1]
            v_int: Intrinsic value estimate [batch, 1]
        """
        features = self.backbone(obs)
        return self.v_extrinsic(features), self.v_intrinsic(features)


class PPOAgent:
    """
    PPO agent with dual value heads and α-controlled reward mixing.

    This is the complete agent that:
    1. Selects actions using the actor
    2. Estimates values using dual critics
    3. Combines intrinsic and extrinsic rewards using α
    4. Updates policy and value functions using PPO

    === THE α MECHANISM (YOUR INDEPENDENT VARIABLE) ===

    The total advantage used for policy updates is:

        A_total = α × A_intrinsic + (1 - α) × A_extrinsic

    where each advantage is computed using GAE (Generalised Advantage
    Estimation) with its own value head and discount factor.

    α = 0: Agent only cares about touch reward (cognitivist/nurture)
    α = 1: Agent only cares about novelty (emergentist/nature)
    α = 0.5: Equal balance between curiosity and task reward
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Networks
        self.actor = Actor(
            config.obs_dim, config.action_dim,
            config.hidden_dim, config.policy_layers
        ).to(self.device)

        self.critic = DualCritic(
            config.obs_dim, config.hidden_dim, config.value_layers
        ).to(self.device)

        # Optimisers
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=config.policy_lr
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=config.value_lr
        )

    def select_action(self, obs: np.ndarray,
                      deterministic: bool = False
                      ) -> Tuple[np.ndarray, float]:
        """Select an action given an observation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(obs_t, deterministic)
        return action.cpu().numpy().squeeze(0), log_prob.item()

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                    dones: np.ndarray, next_value: float,
                    gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalised Advantage Estimation (GAE).

        === WHAT IS GAE? ===
        GAE (Schulman et al., 2016) is a way to estimate "how much better
        was this action than the average?" It balances bias and variance
        using parameter λ:
          - λ = 0: low variance, high bias (1-step TD error)
          - λ = 1: high variance, low bias (Monte Carlo return)
          - λ = 0.95: good balance (standard choice)

        We compute GAE separately for intrinsic and extrinsic rewards,
        then combine using α.

        Args:
            rewards: [T] — rewards at each timestep
            values: [T] — value estimates at each timestep
            dones: [T] — episode termination flags
            next_value: value estimate at T+1
            gamma: discount factor
            lam: GAE λ parameter

        Returns:
            advantages: [T] — advantage estimates
            returns: [T] — target values for the critic
        """
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD error: δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
            # For non-episodic intrinsic rewards, we DON'T mask by done
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]

            # GAE: A_t = δ_t + γλ × A_{t+1}
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_gae
            last_gae = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        PPO update using collected experience.

        === THE FULL UPDATE PIPELINE ===
        1. Compute GAE advantages for both intrinsic and extrinsic rewards
        2. Combine advantages using α
        3. Run PPO clipped objective for multiple epochs
        4. Update both actor and critic

        Args:
            rollout: Dictionary containing:
                - obs: [T, obs_dim]
                - actions: [T, action_dim]
                - log_probs: [T]
                - rewards_ext: [T] — extrinsic (touch) rewards
                - rewards_int: [T] — intrinsic (RND) rewards
                - dones: [T] — episode done flags
                - next_obs: [obs_dim] — final observation

        Returns:
            Dictionary of loss metrics for logging
        """
        cfg = self.config
        device = self.device

        # Convert to tensors
        obs_t = torch.FloatTensor(rollout["obs"]).to(device)
        actions_t = torch.FloatTensor(rollout["actions"]).to(device)
        old_log_probs_t = torch.FloatTensor(
            rollout["log_probs"]
        ).unsqueeze(-1).to(device)

        # Get value estimates for all timesteps
        with torch.no_grad():
            v_ext, v_int = self.critic(obs_t)
            v_ext = v_ext.cpu().numpy().squeeze(-1)
            v_int = v_int.cpu().numpy().squeeze(-1)

            next_obs_t = torch.FloatTensor(
                rollout["next_obs"]
            ).unsqueeze(0).to(device)
            next_v_ext, next_v_int = self.critic(next_obs_t)
            next_v_ext = next_v_ext.item()
            next_v_int = next_v_int.item()

        # ── Compute GAE for EXTRINSIC rewards (episodic) ──
        adv_ext, ret_ext = self.compute_gae(
            rollout["rewards_ext"], v_ext, rollout["dones"],
            next_v_ext, cfg.gamma_ext, cfg.gae_lambda
        )

        # ── Compute GAE for INTRINSIC rewards (non-episodic) ──
        # Note: for intrinsic rewards, we pass dones=zeros because
        # intrinsic returns DON'T reset at episode boundaries.
        # This is a key insight from the RND paper.
        no_dones = np.zeros_like(rollout["dones"])
        adv_int, ret_int = self.compute_gae(
            rollout["rewards_int"], v_int, no_dones,
            next_v_int, cfg.gamma_int, cfg.gae_lambda
        )

        # ── Combine advantages using α ──
        # THIS IS YOUR KEY EXPERIMENTAL VARIABLE
        alpha = cfg.alpha
        combined_advantages = alpha * adv_int + (1 - alpha) * adv_ext

        # Normalise combined advantages (standard PPO practice)
        combined_advantages = (
            (combined_advantages - combined_advantages.mean())
            / (combined_advantages.std() + 1e-8)
        )

        # Convert to tensors
        advantages_t = torch.FloatTensor(combined_advantages).unsqueeze(-1).to(device)
        ret_ext_t = torch.FloatTensor(ret_ext).unsqueeze(-1).to(device)
        ret_int_t = torch.FloatTensor(ret_int).unsqueeze(-1).to(device)

        # ── PPO update epochs ──
        T = obs_t.shape[0]
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(cfg.ppo_epochs):
            # Shuffle and create minibatches
            indices = torch.randperm(T)

            for start in range(0, T, cfg.batch_size):
                end = min(start + cfg.batch_size, T)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_ret_ext = ret_ext_t[mb_idx]
                mb_ret_int = ret_int_t[mb_idx]

                # ── Policy loss (PPO clipped objective) ──
                new_log_probs, entropy = self.actor.evaluate_action(
                    mb_obs, mb_actions
                )
                ratio = (new_log_probs - mb_old_log_probs).exp()

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus — encourages exploration
                entropy_loss = -0.01 * entropy.mean()

                # ── Value loss (both heads) ──
                v_ext_pred, v_int_pred = self.critic(mb_obs)
                value_loss_ext = F.mse_loss(v_ext_pred, mb_ret_ext)
                value_loss_int = F.mse_loss(v_int_pred, mb_ret_int)
                value_loss = 0.5 * (value_loss_ext + value_loss_int)

                # ── Update actor ──
                self.actor_optim.zero_grad()
                (policy_loss + entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()

                # ── Update critic ──
                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = cfg.ppo_epochs * (T // cfg.batch_size + 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "alpha": alpha,
        }
