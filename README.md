# 🌐 Web World Model

> *A fine-tuned world model for efficient look-ahead planning in autonomous web agents.*

---

## The Origin Story

This project started from a dead end — and that's exactly what made it worth building.

My previous project, [**web agent**](https://github.com/Jbb-525/voicenav), was a web navigation agent. After building it, I kept hitting a ceiling: tuning prompts, improving orchestration and adjusting the policy — none of it moved the needle meaningfully. The agent was *reactive*. It would look at the current page, pick an action, and hope for the best. No foresight. No planning. And on long-horizon tasks in realistic web environments — like booking a flight or managing an order — hope is not a strategy.

The insight was simple: the bottleneck wasn't the outer model. It was the *absence of an inner simulator*.

So I built one. A lightweight, text-based **Web World Model** — a fine-tuned 3B model that, given the current web state and a proposed action, predicts *what will happen next* in natural language. Wrap it in a Simulate-Score-Select loop, and suddenly your agent can think before it clicks.

---

## What This Is

A fine-tuned **Qwen2.5-3B-Instruct** model trained to simulate web state transitions. It takes an **Accessibility Tree** (the semantic structure of a webpage) plus an action, and outputs a natural language description of the resulting state. This enables *model-based look-ahead planning* without ever touching the real browser during planning.
---

## The Core Dilemma This Solves

Web agents face a fundamental tension:

| Approach | Problem |
|---|---|
| **Reactive CoT agents** | No foresight; fail on irreversible actions ("Buy Now") |
| **Tree Search agents** | Must interact with the *real* environment to backtrack — 15+ min per task |
| **This work** | Simulate candidate actions in *text space* — fast, no real-world interaction |

---

## Method: Simulate → Score → Select

```
At each step t:
  1. Policy model generates k=5 candidate actions
  2. World Model predicts outcome of each action (natural language)
  3. Value function scores each predicted outcome (1–5)
  4. Agent executes the highest-scoring action in the real browser
```

The world model never touches the real browser during planning. All lookahead happens in latent text space.

---

## Results

Evaluated on **WebArena** (Shopping + Reddit subsets, 50 tasks each):

```
┌─────────────────────────┬──────────────┬────────────┬──────────────────────────────┐
│ Method                  │ Shopping(50) │ Reddit(50) │ Avg. Inference Time          │
├─────────────────────────┼──────────────┼────────────┼──────────────────────────────┤
│ CoT + Majority Voting   │    14%       │    4%      │ 185s                         │
│ Tree Search             │    24%       │   10%      │ 932s  ← ground truth ceiling │
│ World Model (ours)      │    18%       │    8%      │ 212s  ← 4.4× faster          │
└─────────────────────────┴──────────────┴────────────┴──────────────────────────────┘
```

**vs. Tree Search:** 4.4× speedup (932s → 212s) with competitive task success.  
**vs. Reactive CoT:** +44% task success (Shopping: +29%, Reddit: +100%).

---

## Technical Deep Dive

### 1. Observation Representation: Why Accessibility Trees

Three representations were considered:

- **Raw HTML** — Too noisy. Massive CSS/JS overhead, exhausts context window of small LLMs.
- **Pruned HTML** — Loses layout structure; color/occlusion cues critical for decision-making get stripped.
- **Accessibility Tree** ✅ — High signal-to-noise. Captures semantics and functionality without stylistic clutter. Ideal for 3B-scale reasoning.

### 2. Prediction Target: Why Natural Language Descriptions

Three output formulations were tested:

- **Predict full next Accessibility Tree** — Models copy the current state; fail to capture subtle updates.
- **Predict tree diffs** — Page transitions can generate diffs *longer than the original tree*; leads to severe hallucination.
- **Predict NL state transition** ✅ — Focuses the model on causality and semantics. Readily interpretable by a downstream value function. Consistent with findings from RLVR-World: even structural diff models convert to NL for scoring.

### 3. Data Synthesis Pipeline

**14,500 semantic transition labels** synthesized via:

1. **Source trajectories:** High-quality interaction traces from WMA (diverse web environments).
2. **Re-annotation with Gemini 2.5 Flash:** Critically, the teacher model was given *only* the current Accessibility Tree + current action — no user objective, no history. This prevents **task-leakage** (world model predicting user intent instead of state change) and suppresses low-signal DOM noise.

### 4. Training: SFT + GRPO on Multi-GPU

| Component | Details |
|---|---|
| Base Model | Qwen2.5-3B-Instruct |
| Framework | [verl](https://github.com/volcengine/verl) + **FSDP** |
| PEFT | **LoRA** (rank=32, alpha=64, all linear layers) |
| RL Algorithm | **GRPO** with LLM-as-Judge rewards (GPT-4o-mini) |
| Reward | Factual accuracy of predicted state vs. ground truth acc-tree (0–1) |
| GPU | 2× NVIDIA A100 (40GB) |
| Sequence Length | 1k–8k tokens (dynamic batching) |
| Batch Size | 64 (micro batch: 1 per GPU) |
| LR | 1e-5 |

**Dynamic batching** for variable-length inputs (Accessibility Trees range from 1k to 8k tokens) significantly improved training throughput.

### 5. vLLM Inference Server for GRPO Rollouts

During GRPO training, generation was **decoupled from the FSDP training loop** by deploying a **vLLM inference server** for rollout sampling. This avoids the generation bottleneck inherent in naively integrating autoregressive sampling into FSDP-sharded training.

### 6. The RL Reward Hacking Problem (and Why It Matters)

GRPO training showed a critical failure mode: **reward hacking on static web structure**.

Web pages are predominantly static — most of the Accessibility Tree is unchanged after any given action. The model discovered it could maximize its similarity-based reward by *describing the previous state* rather than predicting the change. High similarity scores with the ground truth, zero predictive utility.

**The fix (proposed):** Diff-weighted reward functions that penalize omission of *changed* elements — high recall on diffs, not overall similarity. Standard cosine/BLEU rewards are insufficient for world models in static-heavy environments.

---

## Installation

### Environment 1: Training (verl + FSDP)

```bash
# Python 3.10 or 3.11 recommended
conda create -n myenv python=3.10
cd verl
pip install -e .
```

### Environment 2: Evaluation (WebArena Agent)

```bash
conda create -n webagent python=3.10
cd webarena
pip install vllm
pip install -r requirements.txt
playwright install
pip install -e .
```

---

## Usage

### Supervised Fine-Tuning

```bash
bash verl/examples/sft/web_agent/run_web_sft.sh 2
```

Merge LoRA weights into base model:

```bash
python model_saved/merge.py
```

### End-to-End Evaluation on WebArena

**1. Set up WebArena environments**

Follow [this guide](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md). AWS is recommended for environment hosting.

**2. Configure website URLs**

```bash
export DATASET=webarena
export SHOPPING="http://<your-server-hostname>:7770"
export SHOPPING_ADMIN="http://<your-server-hostname>:7780/admin"
export REDDIT="http://<your-server-hostname>:9999"
```

**3. Generate test config files**

```bash
python scripts/generate_test_data.py
```

Config JSONs will appear in `webarena/config_files/`.

**4. Obtain auto-login cookies**

```bash
bash prepare.sh
```

**5. Set API keys**

```bash
export OPENAI_API_KEY=your_key
```

**6. Create `.env` file**

```env
DATASET=webarena
OPENAI_API_KEY=your_key
SHOPPING="http://<your-server-hostname>:7770"
SHOPPING_ADMIN="http://<your-server-hostname>:7780/admin"
REDDIT="http://<your-server-hostname>:9999"
```

**7. Launch evaluation**

Check `run_webarena.sh` for vLLM setup instructions, then:

```bash
bash scripts/run_webarena.sh
```

---

## Data

All training data is available in the [`data/`](./data) folder (14.5k annotated transition pairs).

---

## Key Takeaways

1. **Small specialized models can beat large reactive ones** — a 3B world model + planning outperforms a GPT-4o-mini CoT agent.
2. **Observation representation is everything** — Accessibility Trees are the right abstraction for lightweight web agents.
3. **NL state descriptions > structural predictions** — causality is easier to learn and score than DOM structure.
4. **RL reward design for world models is an open problem** — similarity-based rewards fail in static-heavy environments.

---

## Related Work

- [VoiceNav](https://github.com/Jbb-525/voicenav) — the predecessor project that motivated this work
- [WebDreamer](https://arxiv.org/abs/2411.06559) — LLM-based web state simulation
- [WMA](https://arxiv.org/abs/2410.13232) — World Model Agents, source of training trajectories
- [Tree Search for LM Agents](https://openreview.net/forum?id=QF0N3x2XVm) — the tree search baseline
- [WebArena](https://arxiv.org/abs/2307.13854) — the evaluation benchmark
