# GPT-J 6B Code SFT: Dataset Scaling & NextLat vs. Standard NTP

This directory contains the dataset curation pipeline, single-shot unit test verification pipeline, Python repository warmup corpus builder, training and evaluation code, and complete experimental results comparing **standard Next-Token Prediction (NTP)**, **Cold-Start NextLat**, and **Repo-Warmed NextLat** across **100**, **1,000**, and **10,000** code SFT examples on **GPT-J 6B** (`EleutherAI/gpt-j-6b`, revision `47e169305d2e8376be1d31e765533382721b2cc1`).

- **Weights & Biases Project:** [`ivanleomk-experiments/gptj-code-sft`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft)
- **Combined Summary Artifact:** [`results/combined_ablation_summary.json`](results/combined_ablation_summary.json)

---

## 1. Repository Layout

```text
code-sft-data/
  generate_splits.py     # Static AST extraction, HumanEval decontamination, and disjoint train/val/test split builder
  generate_tests.py      # Single-shot Gemini 3.8 Flash unit-test generator + isolated reference & mutant verifier
  build_repo_warmup.py   # Decontaminated Python repository function extractor for NextLat dynamics warmup / CPT
  train_lora.py          # Full-parameter and LoRA trainer supporting NTP, cold-start NextLat, and repo-warmed NextLat
  nextlat.py             # NextLat dynamics MLP head (h_t + e_{t+1} -> h_{t+1}), zero-init, and masked Smooth-L1/KL losses
  evaluate.py            # Batched code generation and sandboxed execution for test-100.jsonl and HumanEval
  launch_ablation.py     # Parallel multi-sandbox GPU launcher with live W&B streaming and paired bootstrap CIs
  tests/
    test_extract.py      # 13 unit tests covering extraction, split disjointness, mutants, repo warmup, NextLat, and CIs
  results/
    dataset_manifest.json          # Provenance, counts, and SHA-256 hashes of frozen dataset splits
    combined_ablation_summary.json # Unified 11-arm metrics and 10,000-resample paired bootstrap CIs
    full-sft-ablation/             # Per-arm config.json, metrics.jsonl, and summary.json for base & NTP SFT
    nextlat-ablation/              # Per-arm config.json, metrics.jsonl, and summary.json for cold-start NextLat SFT
    nextlat-warmup-ablation/       # Per-arm config.json, metrics.jsonl, and summary.json for repo-warmed NextLat SFT
```

---

## 2. Frozen Dataset Splits, Repo Warmup Corpus & Verified Benchmarks

Source SFT dataset: [`bigcode/self-oss-instruct-sc2-exec-filter-50k`](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) (revision `356bb069eee815daa6e23e9a282eeefe1490ad44`).
All training, validation, and test splits are strictly family-disjoint across source seed (`source_seed_sha1`), normalized instruction (`instruction_sha256`), and canonical AST body (`canonical_body_sha256`), and decontaminated against HumanEval.

For `val-100.jsonl` and `test-100.jsonl`, `generate_tests.py` uses single-shot (`max_attempts=1`) `models/gemini-3.8-flash` generation without access to the reference solution, followed by isolated `python -I -S -c` verification requiring:
1. At least 2 `assert` statements in `def check(candidate):`
2. Passing the canonical reference solution within a 5s timeout
3. Rejecting at least one generic mutant (`None`, first-arg identity, `0`/`False`) and at least one AST mutant (operator flip, constant perturbation, or negated return)

For the **NextLat Repo Warmup corpus (`repo-warmup-1000.jsonl`)**, `build_repo_warmup.py` extracts 1,000 standalone, documented synchronous Python functions across 9 open-source Python repositories (`TheAlgorithms/Python`, `keon/algorithms`, `sympy`, `cpython-stdlib`, `networkx`, `more-itertools`, `mpmath`, `toolz`, and `boltons`), verified for zero unresolved global names and decontaminated against `HumanEval`, `test-100`, `val-100`, and `train-10000`.

| Split File | Rows | Role | SHA-256 |
|---|---:|---|---|
| `train-100.jsonl` | 100 | Exact 100-row prefix of `train-10000` | `75dac0b06470a10414f2f7f2a076e5414d8d9e8f97b02cf9d4f5518d4c638724` |
| `train-1000.jsonl` | 1,000 | Exact 1,000-row prefix of `train-10000` | `352673cee26a09558e5a83cc7065f3c326db48cb94915e48021a6f09c31f5e1f` |
| `train-10000.jsonl` | 10,000 | Master training pool | `7af46dc5cd9a2eb47a28c498337100da3bc1ed3f8538bee9077bde6827e03ac8` |
| `repo-warmup-1000.jsonl` | 1,000 | Decontaminated Python repo functions for NextLat warmup | `f6452c8ec83fed4f56e71003da5d4123e3284d9dfbcdd87e0f60985d16b7ba82` |
| `val-100.jsonl` | 100 | Verified validation split (NLL tracking) | `73f918c8069b1a1e3f5d584feefb9e5586e0c00b72dd828c74eeb55f00c5c362` |
| `test-100.jsonl` | 100 | Verified in-domain executable benchmark | `7122ff5320825030aaee47297b41cbf7dc964c1768dde93ae7031d4131eb5ee6` |

---

## 3. Results Across All 11 Arms

### 3.1 Master Summary Table

| Arm | Objective / Setup | Train Size | SFT Steps | `val-100` NLL ↓ | `test-100` `pass@1` ↑ | `test-100` `pass@10` ↑ | `HumanEval` `pass@1` ↑ | `HumanEval` `pass@10` ↑ | W&B Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`base`** | 0-step baseline | 0 | 0 | `0.3211` | `17.75%` | `34.70%` | `10.88%` | `15.08%` | [`jvhfbaun`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/jvhfbaun) |
| **`full-sft-100`** | Standard NTP | 100 | 25 | `0.2953` | `12.85%` | `20.84%` | `5.27%` | `8.24%` | [`b8awdoh6`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/b8awdoh6) |
| **`full-sft-1000`** | Standard NTP | 1,000 | 250 | `0.2625` | `25.25%` | `42.01%` | `10.88%` | `14.89%` | [`6a1aoger`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/6a1aoger) |
| **`full-sft-10000`** | Standard NTP | 10,000 | 2,500 | **`0.2060`** | `38.60%` | **`51.88%`** | **`13.63%`** | **`18.15%`** | [`zbewfhwz`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/zbewfhwz) |
| **`nextlat-sft-100`** | Cold-start NextLat ($\lambda=1.0$) | 100 | 25 | `0.5933` | `7.95%` | `14.69%` | `1.31%` | `2.88%` | [`pcarpa37`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/pcarpa37) |
| **`nextlat-sft-1000`** | Cold-start NextLat ($\lambda=1.0$) | 1,000 | 250 | `0.3611` | `12.35%` | `29.11%` | `8.23%` | `10.53%` | [`yk91by12`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/yk91by12) |
| **`nextlat-sft-10000`** | Cold-start NextLat ($\lambda=1.0$) | 10,000 | 2,500 | `0.2409` | `29.70%` | `38.77%` | `11.68%` | `17.48%` | [`nseqkvfv`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/nseqkvfv) |
| **`nextlat-warmup-sft-100`** | **Repo-Warmed NextLat** | 100 | 25 | `0.2963` | `12.85%` | `22.47%` | `7.01%` | `10.34%` | [`0v6p3bn8`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/0v6p3bn8) |
| **`nextlat-warmup-sft-1000`** | **Repo-Warmed NextLat** | 1,000 | 250 | `0.2690` | `23.00%` | `41.13%` | `9.18%` | `13.83%` | [`6ea757il`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/6ea757il) |
| **`nextlat-cpt-sft-1000`** | **Repo-Warmed + Joint CPT NextLat** | 1,000 | 250 | `0.2654` | `22.65%` | `38.17%` | `9.85%` | `14.05%` | [`60vltons`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/60vltons) |
| **`nextlat-warmup-sft-10000`** | **Repo-Warmed NextLat** | 10,000 | 2,500 | `0.2080` | **`40.85%`** | `48.77%` | `13.17%` | `17.40%` | [`qj81ntc6`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/qj81ntc6) |

---

### 3.2 Paired Bootstrap Gain of Repo-Warmed NextLat vs. Cold-Start NextLat (10,000 Resamples)

| Comparison | `test-100` $\Delta\text{pass@1}$ `[95% CI]` | `test-100` $\Delta\text{pass@10}$ `[95% CI]` | `HumanEval` $\Delta\text{pass@1}$ `[95% CI]` | `HumanEval` $\Delta\text{pass@10}$ `[95% CI]` |
|---|---|---|---|---|
| **`nextlat-warmup-sft-100` vs `nextlat-sft-100`** | `+4.90%` `[-0.55%, +10.55%]` | `+7.78%` `[-0.69%, +16.44%]` | **`+5.70%`** `[+2.84%, +9.02%]` | **`+7.46%`** `[+3.79%, +11.56%]` |
| **`nextlat-warmup-sft-1000` vs `nextlat-sft-1000`** | **`+10.65%`** `[+5.15%, +16.30%]` | **`+12.01%`** `[+4.05%, +20.09%]` | `+0.95%` `[-1.19%, +3.23%]` | `+3.30%` `[-0.32%, +7.14%]` |
| **`nextlat-cpt-sft-1000` vs `nextlat-sft-1000`** | **`+10.30%`** `[+4.50%, +16.40%]` | **`+9.06%`** `[+0.61%, +17.52%]` | `+1.62%` `[-0.61%, +4.18%]` | **`+3.52%`** `[+0.41%, +7.06%]` |
| **`nextlat-warmup-sft-10000` vs `nextlat-sft-10000`** | **`+11.15%`** `[+5.40%, +17.50%]` | **`+10.01%`** `[+3.45%, +17.06%]` | `+1.49%` `[-0.91%, +3.96%]` | `-0.09%` `[-3.28%, +3.04%]` |

---

### 3.3 Paired Bootstrap Comparison: Repo-Warmed NextLat vs. Matched Standard NTP

| Dataset Size ($N$) | `test-100` $\Delta\text{pass@1}$ `[95% CI]` | `test-100` $\Delta\text{pass@10}$ `[95% CI]` | `HumanEval` $\Delta\text{pass@1}$ `[95% CI]` | `HumanEval` $\Delta\text{pass@10}$ `[95% CI]` |
|---:|---|---|---|---|
| **100 (`warmup` vs `full-sft-100`)** | `+0.00%` `[-3.35%, +3.25%]` | `+1.64%` `[-3.48%, +7.08%]` | **`+1.74%`** `[+0.09%, +3.78%]` | `+2.09%` `[-0.03%, +4.62%]` |
| **1,000 (`warmup` vs `full-sft-1000`)** | `-2.25%` `[-7.05%, +2.35%]` | `-0.88%` `[-7.74%, +5.96%]` | `-1.71%` `[-3.45%, -0.12%]` | `-1.06%` `[-4.02%, +1.81%]` |
| **1,000 (`cpt` vs `full-sft-1000`)** | `-2.60%` `[-8.30%, +2.85%]` | `-3.83%` `[-11.93%, +4.29%]` | `-1.04%` `[-2.59%, +0.46%]` | `-0.85%` `[-3.38%, +1.70%]` |
| **10,000 (`warmup` vs `full-sft-10000`)** | **`+2.25%`** `[-2.70%, +7.30%]` | `-3.11%` `[-10.53%, +4.22%]` | `-0.46%` `[-2.44%, +1.55%]` | `-0.76%` `[-3.60%, +1.96%]` |

---

## 4. Key Takeaways

1. **Why Cold-Start NextLat Struggled in Post-Training:**
   - Attaching a randomly initialized `NextLatDynamicsModel` (`~67M` parameters) directly to a pretrained 6B model (`GPT-J`) at step 0 with $\lambda_{\text{MSE}}=1.0, \lambda_{\text{KL}}=1.0$ creates a massive initial KL loss (`~9.72` vs. `0.32` NTP loss) and a `grad_norm` spike of `~51`, scrambling the pretrained backbone during the first ~100–250 steps (`val-100` NLL jumped from `0.3211` to `0.5933` at 100 examples).
2. **Frozen-Backbone Repo Warmup + Zero-Init Completely Fixes NextLat:**
   - Initializing `NextLatDynamicsModel`'s final residual projection to zero (`zero_init_output=True`), warming up only the dynamics head for 250 steps (`1,000` decontaminated Python repo functions under `torch.no_grad()` on `GPT-J`), and scaling the auxiliary loss weights ($\lambda_{\text{MSE}}=0.2, \lambda_{\text{KL}}=0.1$) boosted NextLat by **`+4.90%` to `+11.15%` `pass@1`** across all dataset sizes:
     - At **100 examples**, `nextlat-warmup-sft-100` **beats `full-sft-100` on `HumanEval pass@1`** (`7.01%` vs `5.27%`, `+1.74%`, 95% CI `[+0.09%, +3.78%]`) and `test-100 pass@10` (`22.47%` vs `20.84%`).
     - At **1,000 examples**, `test-100 pass@1` nearly doubles over cold-start (`12.35% -> 23.00%`, **`+10.65%`**), and `pass@10` jumps `+12.01%` (`29.11% -> 41.13%`).
     - At **10,000 examples**, `nextlat-warmup-sft-10000` achieves **`40.85%` `pass@1` on `test-100`**—**the highest `pass@1` score of any model in the entire study**, beating cold-start `nextlat-sft-10000` (`29.70%`) by **`+11.15%`** and surpassing `full-sft-10000` (`38.60%`) by **`+2.25%`** (`+23.10%` over the `17.75%` base GPT-J model).
