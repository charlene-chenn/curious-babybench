# Self-Touch World Model Project — Step-by-Step Walkthrough

## How to Read This Guide

Each step below tells you:
- **What you're building** (the component)
- **Why it matters for your project** (coursework relevance)
- **Where it sits on the frontier** (world model research context)
- **Which file to look at** (the code)

---

## Step 0: The Big Picture

You are building an RL agent that learns to touch its own body in BabyBench.
The twist: instead of just maximising a touch reward, you're studying how the
*balance* between curiosity (intrinsic motivation via RND) and task reward
(extrinsic touch signal) affects:

1. How quickly the agent discovers self-touch
2. How diverse its touch behavior is (which body parts it reaches)
3. How good its internal *world model* becomes

This directly maps to the **cognitivist vs emergentist** debate in developmental AI:
- Cognitivist = "nurture" = extrinsic reward guides behavior toward a goal
- Emergentist = "nature" = intrinsic curiosity lets behavior emerge from exploration

Your α parameter controls the dial between these two.

---

## Step 1: Configuration (`config.py`)

**What:** A single place for all hyperparameters.

**Why for your project:** When you run experiments across different α values,
you want to change ONE number and re-run. Clean config also makes your video
explanation easier — you can show the config and say "here's what I varied."

**Frontier context:** DreamerV3's key innovation was using *fixed hyperparameters*
across 150+ tasks (Hafner et al., 2025). They achieved this through normalisation
tricks (symlog, percentile return normalisation). We borrow that philosophy:
our config should work for both self-touch and hand-regard without tuning.

**Key parameters to understand:**
- `alpha`: The nature/nurture balance (your independent variable)
- `rnd_output_dim`: Dimensionality of the RND embedding space
- `latent_dim`: Size of the world model's compressed representation
- `dual_value_heads`: Whether to use separate value functions for intrinsic
  and extrinsic rewards (from the RND paper, Burda et al. 2018)

---

## Step 2: The World Model (`world_model.py`)

**What:** A neural network that learns a compressed representation of the
environment and can predict what happens next.

**Why for your project:** This is the "world model" in your title. The agent
doesn't just react to observations — it builds an internal model of its body
and the physics of self-touch. The quality of this model is one of your
dependent variables.

**Frontier context:** This is inspired by DreamerV3's RSSM (Recurrent State-Space
Model). The full RSSM has:
- An **encoder** that compresses observations into latent states
- A **dynamics predictor** (recurrent) that predicts next latent state from
  current state + action (this is the "imagination" engine)
- A **decoder** that reconstructs observations from latent states
- A **reward predictor** that estimates reward from latent states

In frontier systems like DreamerV3, the actor-critic is trained *entirely in
imagination* — rollouts happen in latent space, not in the real environment.
This is vastly more sample-efficient because you can imagine thousands of
trajectories from a single real experience.

**Our simplification:** For a coursework project, we implement a lighter version:
- Encoder (observation → latent z_t)
- Dynamics predictor (z_t + a_t → z_{t+1})  [simplified, no recurrence]
- Decoder (z_t → reconstructed observation)
- Reward predictor (z_t → predicted reward)

The reconstruction loss tells you how good the world model is — this is a
key metric for your results section.

**Key concepts in the code:**
- `encode()`: Maps raw sensor data to a compact latent vector
- `predict_next()`: The "imagination" step — given current state + action,
  what happens next?
- `decode()`: Reconstructs what the agent "thinks" it should be seeing
- `compute_loss()`: Trains the world model on real experience

---

## Step 3: Random Network Distillation (`rnd.py`)

**What:** A module that produces *intrinsic reward* — a curiosity signal that
rewards the agent for visiting states it hasn't seen before.

**Why for your project:** RND is your "nature" signal. Without it, the agent
only learns what you explicitly reward (touching). With it, the agent is
driven to explore novel states, potentially discovering self-touch on its own
as a byproduct of curiosity about its body.

**Frontier context:** RND (Burda et al., 2018) was a breakthrough in exploration
for sparse-reward RL. The key insight is elegant:

1. Take a randomly initialised neural network (the **target**). Freeze it.
2. Train a second network (the **predictor**) to match the target's output.
3. The prediction error IS the intrinsic reward.

Why does this work? For states the agent visits often, the predictor learns
to match the target well → low error → low reward. For novel states, the
predictor hasn't been trained on them → high error → high reward. The agent
is thus incentivised to seek out states it hasn't seen before.

**Connection to your coursework narrative:**
- High RND error = "the agent doesn't understand this part of its body yet"
- Low RND error = "the agent has a good model of this sensation"
- Tracking RND error over training shows the agent "learning about itself"

**Important implementation detail — observation normalisation:**
RND is sensitive to the scale of observations. We use a running mean/std
normaliser (from the original paper) to keep inputs stable. Without this,
the predictor can trivially reduce error by exploiting scale differences
rather than genuinely learning about novel states.

---

## Step 4: The Agent (`agent.py`)

**What:** A PPO (Proximal Policy Optimisation) agent with dual value heads
and the reward combiner.

**Why for your project:** This is where nature meets nurture. The agent has:
- A **policy** (actor) that decides what joint torques to apply
- Two **value heads** (critic): one for intrinsic reward, one for extrinsic
- A **reward combiner** that mixes r_intrinsic and r_extrinsic using α

**Frontier context:**

*Why PPO?* PPO (Schulman et al., 2017) is the workhorse of modern RL. It's
stable, scales well, and is the base algorithm in DreamerV3's actor-critic.
The key idea: update the policy, but clip the update so you don't change too
much at once. This prevents the catastrophic forgetting that plagued earlier
policy gradient methods.

*Why dual value heads?* This comes directly from the RND paper. Intrinsic and
extrinsic rewards have fundamentally different temporal properties:
- Extrinsic reward is **episodic**: it resets each episode
- Intrinsic reward is **non-episodic**: a state that's novel now will
  always have been novel in that moment, regardless of episode boundaries

Using a single value function to estimate both leads to poor estimates of
both. Separate heads let each value function learn at its own timescale.

*The reward combiner:*
```
r_total = α × r_intrinsic + (1 - α) × r_extrinsic
```

This is your key experimental lever. By sweeping α from 0 to 1, you map
the spectrum from pure cognitivist (α=0, only touch reward) to pure
emergentist (α=1, only curiosity).

**The adaptive α stretch goal:**
Instead of a fixed α, you could make it a function of the RND signal:
- When mean RND error is high → the agent is in unfamiliar territory → lean
  on intrinsic motivation to explore more
- When mean RND error is low → the agent understands its environment → lean
  on extrinsic reward to exploit what it knows

This is the "self-improvement loop": the world model's uncertainty drives
exploration, which improves the world model, which reduces uncertainty.

---

## Step 5: Training (`train.py`)

**What:** The main training loop that ties everything together.

**Why for your project:** This is what you actually run. It:
1. Initialises BabyBench with the self-touch environment
2. Collects experience (observations, actions, rewards, touch events)
3. Trains the world model on real experience
4. Computes intrinsic rewards via RND
5. Combines rewards using α
6. Updates the policy via PPO
7. Logs everything for your results section

**Frontier context:**

*Experience collection:* In DreamerV3, the training loop alternates between
"acting in the real environment" and "dreaming" (training in imagination).
Our simplified version trains the policy on real experience, but uses the
world model's prediction quality as a metric rather than for imagination.

*Logging:* We track metrics that map directly to your video narrative:
- `touch_count`: How often the agent touches itself (primary behavior metric)
- `body_parts_touched`: Which parts (diversity metric)
- `world_model_loss`: How well the agent understands its body (model quality)
- `rnd_reward_mean`: How novel the states are (exploration metric)
- `extrinsic_reward`: The raw touch signal

---

## Step 6: Evaluation (`evaluate.py`)

**What:** Scripts to analyse results and produce figures for your video.

**Why for your project:** The evaluation criteria say "results need to fit
narrative" — you need clear plots showing:
1. Learning curves per α condition
2. Body-part touch heatmaps
3. World model accuracy over time
4. Comparison table across conditions

**Frontier context:** The BabyBench evaluation protocol measures self-touch
via touch sensor activations on MIMo's body. Our evaluation extends this
with world model quality metrics, connecting developmental behavior (touch)
to internal representation quality (world model) — a bridge between
behavioral and computational neuroscience perspectives.

---

## Step 7: Running Experiments

### Quick start (single run)
```bash
cd self_touch_project
python train.py --alpha 0.5 --episodes 500 --seed 42
```

### Full experiment sweep
```bash
for alpha in 0.0 0.25 0.5 0.75 1.0; do
    for seed in 1 2 3; do
        python train.py --alpha $alpha --seed $seed --episodes 500
    done
done
```

### Generate results
```bash
python evaluate.py --results_dir results/
```

---

## How This Maps to Your 3-Minute Video

### Minute 1: Motivation & References
- Show the nature vs nurture framing
- Reference RND (Burda 2018) and DreamerV3 (Hafner 2025)
- "We ask: can an infant-like agent discover self-touch through
  curiosity alone, or does it need guidance?"

### Minute 2: Architecture & Implementation
- Show the architecture diagram (from our earlier conversation)
- Walk through: observations → world model → RND → reward combiner → policy
- Use terms: action space, observation space, intrinsic/extrinsic reward,
  dual value heads, latent representation

### Minute 3: Results & Discussion
- Learning curves: "Balanced α outperformed pure approaches because..."
- Touch heatmaps: "The curious agent explored more body parts..."
- World model quality: "Better exploration led to better internal models..."
- Limitations: "With more compute, we could use full DreamerV3 imagination..."
- Impact: "This suggests developmental AI benefits from both innate drives
  and environmental feedback, just as human infants do."
