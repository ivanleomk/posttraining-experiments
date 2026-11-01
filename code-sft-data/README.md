# GPT-J 6B Code SFT: Dataset Scaling & NextLat vs. Standard NTP

This directory contains the dataset curation pipeline, single-shot unit test verification pipeline, training and evaluation code, and complete experimental results comparing **standard Next-Token Prediction (NTP)** against **NextLat (Next-Latent Prediction)** across **100**, **1,000**, and **10,000** code SFT examples on **GPT-J 6B** (`EleutherAI/gpt-j-6b`, revision `47e169305d2e8376be1d31e765533382721b2cc1`).

- **Weights & Biases Project:** [`ivanleomk-experiments/gptj-code-sft`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft)
- **Combined Summary Artifact:** [`results/combined_ablation_summary.json`](results/combined_ablation_summary.json)

---

## 1. Repository Layout

```text
code-sft-data/
  generate_splits.py   # Static AST extraction, HumanEval decontamination, and disjoint train/val/test split builder
  generate_tests.py    # Single-shot Gemini 3.8 Flash unit-test generator + isolated reference & mutant verifier
  train_lora.py        # Full-parameter and LoRA SFT trainer supporting both standard NTP and NextLat objectives
  nextlat.py           # NextLat dynamics MLP head (h_t + e_{t+1} -> h_{t+1}) + masked Smooth-L1 & KL losses
  evaluate.py          # Batched code generation and sandboxed execution for test-100.jsonl and HumanEval
  launch_ablation.py   # Parallel multi-sandbox GPU launcher with live W&B streaming and paired bootstrap CIs
  tests/
    test_extract.py    # Unit tests covering extraction, split disjointness, mutant checks, NextLat masks, and CIs
  results/
    dataset_manifest.json          # Provenance, counts, and SHA-256 hashes of frozen dataset splits
    combined_ablation_summary.json # Unified 7-arm metrics and 10,000-resample paired bootstrap CIs
    full-sft-ablation/             # Per-arm config.json, metrics.jsonl, and summary.json for base & NTP SFT
    nextlat-ablation/              # Per-arm config.json, metrics.jsonl, and summary.json for NextLat SFT
```

---

## 2. Frozen Dataset Splits & Verified Benchmarks

Source dataset: [`bigcode/self-oss-instruct-sc2-exec-filter-50k`](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) (revision `356bb069eee815daa6e23e9a282eeefe1490ad44`).
All training, validation, and test splits are strictly family-disjoint across source seed (`source_seed_sha1`), normalized instruction (`instruction_sha256`), and canonical AST body (`canonical_body_sha256`), and decontaminated against HumanEval.

For `val-100.jsonl` and `test-100.jsonl`, `generate_tests.py` uses single-shot (`max_attempts=1`) `models/gemini-3.8-flash` generation without access to the reference solution, followed by isolated `python -I -S -c` verification requiring:
1. At least 2 `assert` statements in `def check(candidate):`
2. Passing the canonical reference solution within a 5s timeout
3. Rejecting at least one generic mutant (`None`, first-arg identity, `0`/`False`) and at least one AST mutant (operator flip, constant perturbation, or negated return)

| Split File | Rows | Role | SHA-256 |
|---|---:|---|---|
| `train-100.jsonl` | 100 | Exact 100-row prefix of `train-10000` | `75dac0b06470a10414f2f7f2a076e5414d8d9e8f97b02cf9d4f5518d4c638724` |
| `train-1000.jsonl` | 1,000 | Exact 1,000-row prefix of `train-10000` | `352673cee26a09558e5a83cc7065f3c326db48cb94915e48021a6f09c31f5e1f` |
| `train-10000.jsonl` | 10,000 | Master training pool | `7af46dc5cd9a2eb47a28c498337100da3bc1ed3f8538bee9077bde6827e03ac8` |
| `val-100.jsonl` | 100 | Verified validation split (NLL tracking) | `73f918c8069b1a1e3f5d584feefb9e5586e0c00b72dd828c74eeb55f00c5c362` |
| `test-100.jsonl` | 100 | Verified in-domain executable benchmark | `7122ff5320825030aaee47297b41cbf7dc964c1768dde93ae7031d4131eb5ee6` |

---

## 3. Experimental Setup

Every training arm starts from the original `EleutherAI/gpt-j-6b` checkpoint (`6B` parameters):
- **Training mode:** Full-parameter fine-tuning (`--mode full`) in FP32 master weights with BF16 mixed-precision autocast and gradient checkpointing.
- **Optimizer & schedule:** `bitsandbytes.optim.AdamW8bit` (`lr = 1e-5`, `weight_decay = 0.01`, cosine schedule with 5% warmup, global grad-norm clipping at `1.0`).
- **Batching:** 1 epoch, microbatch size 1, gradient accumulation 4 (effective batch size 4 sequences; token-weighted gradient accumulation across answer tokens + EOS; prompt tokens masked with `-100`).
- **NextLat auxiliary objective (`--objective nextlat`):**
  - Following [`JaydenTeoh/NextLat`](https://github.com/JaydenTeoh/NextLat) (`3770be6009cea2b3c455a9ce7f2ca88b504bb955`), a 3-layer residual `NextLatDynamicsModel` (`proj_factor = 0.5`, `hidden_dim = 4096`, `~67M` parameters) predicts the next final-layer hidden state $\hat{h}_{t+1} = h_t + \text{MLP}(\text{LN}([e_{t+1}; h_t]))$.
  - Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \lambda_{\text{MSE}}\,\mathcal{L}_{\text{SmoothL1}}(\hat{h}_{t+1}, \text{sg}(h_{t+1})) + \lambda_{\text{KL}}\,\text{KL}(\text{sg}(W_{\text{lm}} h_{t+1}) \,\|\, \text{sg}(W_{\text{lm}})\hat{h}_{t+1})$ with $\lambda_{\text{MSE}} = 1.0, \lambda_{\text{KL}} = 1.0$, masked strictly to in-answer non-EOS transitions.
  - At inference time, `NextLatDynamicsModel` is discarded and standard autoregressive decoding is used.
- **Evaluation protocol:** 20 samples per task (`temperature = 0.2`, `top_p = 0.95`, `max_new_tokens = 512`, `seed = 42`) on both `test-100.jsonl` (100 tasks) and `HumanEval` (164 tasks), reporting unbiased `pass@1` and `pass@10` estimators plus 10,000-resample paired task-level bootstrap 95% confidence intervals.

---

## 4. Results

### 4.1 Summary Table Across All 7 Arms

| Arm | Objective | Train Size | Steps | `val-100` NLL ↓ | `test-100` `pass@1` ↑ | `test-100` `pass@10` ↑ | `HumanEval` `pass@1` ↑ | `HumanEval` `pass@10` ↑ | W&B Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`base`** | — | 0 | 0 | `0.3211` | `17.75%` | `34.70%` | `10.88%` | `15.08%` | [`jvhfbaun`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/jvhfbaun) |
| **`full-sft-100`** | NTP | 100 | 25 | `0.2953` | `12.85%` | `20.84%` | `5.27%` | `8.24%` | [`b8awdoh6`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/b8awdoh6) |
| **`full-sft-1000`** | NTP | 1,000 | 250 | `0.2625` | `25.25%` | `42.01%` | `10.88%` | `14.89%` | [`6a1aoger`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/6a1aoger) |
| **`full-sft-10000`** | NTP | 10,000 | 2,500 | **`0.2060`** | **`38.60%`** | **`51.88%`** | **`13.63%`** | **`18.15%`** | [`zbewfhwz`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/zbewfhwz) |
| **`nextlat-sft-100`** | NextLat | 100 | 25 | `0.5933` | `7.95%` | `14.69%` | `1.31%` | `2.88%` | [`pcarpa37`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/pcarpa37) |
| **`nextlat-sft-1000`** | NextLat | 1,000 | 250 | `0.3611` | `12.35%` | `29.11%` | `8.23%` | `10.53%` | [`yk91by12`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/yk91by12) |
| **`nextlat-sft-10000`** | NextLat | 10,000 | 2,500 | `0.2409` | `29.70%` | `38.77%` | `11.68%` | `17.48%` | [`nseqkvfv`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/nseqkvfv) |

### 4.2 Paired Bootstrap Differences vs. `base` (10,000 Resamples)

| Arm | `test-100` $\Delta\text{pass@1}$ `[95% CI]` | `test-100` $\Delta\text{pass@10}$ `[95% CI]` | `HumanEval` $\Delta\text{pass@1}$ `[95% CI]` | `HumanEval` $\Delta\text{pass@10}$ `[95% CI]` |
|---|---|---|---|---|
| **`full-sft-100`** | `-4.90%` `[-11.30%, +1.25%]` | `-13.86%` `[-22.74%, -5.06%]` | `-5.61%` `[-9.36%, -2.10%]` | `-6.84%` `[-11.49%, -2.47%]` |
| **`full-sft-1000`** | **`+7.50%`** `[+1.45%, +13.70%]` | `+7.31%` `[-1.08%, +15.86%]` | `+0.00%` `[-3.29%, +3.23%]` | `-0.19%` `[-3.84%, +3.52%]` |
| **`full-sft-10000`** | **`+20.85%`** `[+13.60%, +27.85%]` | **`+17.19%`** `[+8.47%, +26.15%]` | `+2.74%` `[-0.85%, +6.65%]` | `+3.07%` `[-1.42%, +7.77%]` |
| **`nextlat-sft-100`** | `-9.80%` `[-16.30%, -3.70%]` | `-20.00%` `[-28.72%, -11.37%]` | `-9.57%` `[-13.81%, -5.82%]` | `-12.20%` `[-17.28%, -7.33%]` |
| **`nextlat-sft-1000`** | `-5.40%` `[-10.70%, -0.35%]` | `-5.58%` `[-13.94%, +2.85%]` | `-2.65%` `[-5.88%, +0.34%]` | `-4.56%` `[-8.08%, -1.37%]` |
| **`nextlat-sft-10000`** | **`+11.95%`** `[+4.00%, +19.95%]` | `+4.07%` `[-5.48%, +13.37%]` | `+0.79%` `[-2.35%, +3.90%]` | `+2.40%` `[-2.22%, +7.12%]` |

### 4.3 Paired Bootstrap Differences: `NextLat` vs. Matched `NTP` (`nextlat-sft-N` minus `full-sft-N`)

| Dataset Size ($N$) | `test-100` $\Delta\text{pass@1}$ `[95% CI]` | `test-100` $\Delta\text{pass@10}$ `[95% CI]` | `HumanEval` $\Delta\text{pass@1}$ `[95% CI]` | `HumanEval` $\Delta\text{pass@10}$ `[95% CI]` |
|---:|---|---|---|---|
| **100** | `-4.90%` `[-10.10%, -0.05%]` | `-6.14%` `[-13.66%, +1.37%]` | `-3.96%` `[-6.80%, -1.62%]` | `-5.37%` `[-9.30%, -1.86%]` |
| **1,000** | `-12.90%` `[-19.60%, -6.60%]` | `-12.89%` `[-20.47%, -5.47%]` | `-2.65%` `[-5.58%, -0.06%]` | `-4.37%` `[-7.99%, -1.14%]` |
| **10,000** | `-8.90%` `[-16.30%, -2.05%]` | `-13.11%` `[-20.60%, -5.97%]` | `-1.95%` `[-4.36%, +0.40%]` | `-0.67%` `[-3.75%, +2.44%]` |

---

## 5. Key Takeaways & Analysis

1. **Data Scaling Law for Full-Parameter Code SFT:**
   - Very small SFT (`100` examples / `25` steps) slightly degrades functional accuracy (`-4.90%` on `test-100`, `-5.61%` on `HumanEval`) despite lowering validation NLL (`0.3211 -> 0.2953`), as initial AdamW updates perturb pretrained representations before distribution alignment completes.
   - By `1,000` examples (`250` steps), in-domain `test-100 pass@1` improves by **`+7.50%`** (`17.75% -> 25.25%`) and `HumanEval` recovers to baseline (`10.88%`).
   - By `10,000` examples (`2,500` steps), `test-100 pass@1` more than doubles (**`17.75% -> 38.60%`**, `+20.85%`) and out-of-distribution `HumanEval` reaches **`13.63%` `pass@1`** (`+2.74%`) and **`18.15%` `pass@10`** (`+3.07%`).

2. **NextLat Cold-Start Dynamics on a Pretrained Backbone:**
   - In the original NextLat paper (`JaydenTeoh/NextLat`), the transformer backbone and `NextLatDynamicsModel` are trained **jointly from scratch**.
   - When attaching a randomly initialized `NextLatDynamicsModel` (`~67M` parameters, $\mathcal{N}(0, 0.02)$ initialization) to a **pretrained** 6B backbone (`GPT-J`) with $\lambda_{\text{MSE}} = 1.0, \lambda_{\text{KL}} = 1.0$, the initial auxiliary KL loss at step 1 is `~9.72` (nearly $30\times$ the `0.32` NTP loss), producing a `grad_norm` spike of `~51` during the first ~25 optimizer steps while the dynamics MLP learns GPT-J's latent geometry.
   - Consequently, at `100` examples (`25` steps) and `1,000` examples (`250` steps), NextLat is still paying off this early cold-start shock (`val-100` NLL peaks at `0.5933` around step 25 before falling to `0.3611` at step 250).
   - Once given sufficient optimization budget at **`10,000` examples (`2,500` steps)**, the auxiliary losses converge (`nextlat_mse_loss = 0.1243`, `nextlat_kl_loss = 0.3195`), `val-100` NLL drops to **`0.2409`**, `test-100 pass@1` jumps to **`29.70%`** (`+11.95%` over `base`), and `HumanEval pass@10` reaches **`17.48%`** (statistically indistinguishable from `full-sft-10000` at `18.15%`, $\Delta = -0.67\%$, 95% CI `[-3.75%, +2.44%]`).
   - **Remedies for Cold-Start NextLat Post-Training:**
     1. **Zero-init residual output (`zero_init_output=True` in `nextlat.py`):** Initializing the final linear layer of `NextLatDynamicsModel` to zero makes $\hat{h}_{t+1} = h_t$ at step 0, cutting initial KL loss by ~3x.
     2. **Two-stage dynamics warmup (or continued code NTP + NextLat):** Freezing the GPT-J backbone for the first `500–1,000` steps (around `2,000–4,000` code completions) so `NextLatDynamicsModel` aligns with the pretrained latent space before unfreezing the backbone for joint SFT.

---

## 6. Reproducing

Run unit tests locally:

```bash
python -m unittest discover -s code-sft-data/tests -v
```

Launch a full multi-sandbox ablation on CoreWeave GPU sandboxes with live W&B logging:

```bash
# Standard NTP full-SFT ablation (base, 100, 1000, 10000)
python code-sft-data/launch_ablation.py \
  --objective ntp \
  --splits-dir /path/to/splits-v1 \
  --output-dir /path/to/runs/full-sft-ablation

# NextLat full-SFT ablation (100, 1000, 10000)
python code-sft-data/launch_ablation.py \
  --objective nextlat \
  --arms nextlat-sft-100 --arms nextlat-sft-1000 --arms nextlat-sft-10000 \
  --splits-dir /path/to/splits-v1 \
  --output-dir /path/to/runs/nextlat-ablation
```
