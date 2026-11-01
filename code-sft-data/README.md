# GPT-J 6B Code SFT: Dataset Scaling, NextLat vs. Standard NTP & 2.5x Speculative Decoding

This directory contains the dataset curation pipeline, single-shot unit test verification pipeline, Python repository warmup corpus builder, training and evaluation code, and complete experimental results comparing **standard Next-Token Prediction (NTP)**, **Cold-Start NextLat**, and **Repo-Warmed NextLat** across **100**, **1,000**, and **10,000** code SFT examples on **GPT-J 6B** (`EleutherAI/gpt-j-6b`, revision `47e169305d2e8376be1d31e765533382721b2cc1`), as well as **lossless self-speculative decoding** using the trained `67M`-parameter `NextLatDynamicsModel` on both `test-100` and `HumanEval`.

- **Weights & Biases Project:** [`ivanleomk-experiments/gptj-code-sft`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft)
- **Combined Summary Artifact:** [`results/combined_ablation_summary.json`](results/combined_ablation_summary.json)

---

## 1. Repository Layout

```text
code-sft-data/
  generate_splits.py     # Static AST extraction, HumanEval decontamination, and disjoint train/val/test split builder
  generate_tests.py      # Single-shot Gemini 3.8 Flash unit-test generator + isolated reference & mutant verifier
  build_repo_warmup.py   # Decontaminated Python repository function extractor for NextLat dynamics warmup / CPT
  train_lora.py          # Full-parameter and LoRA trainer supporting NTP, cold-start NextLat, repo-warmed NextLat, and spec-decode
  nextlat.py             # NextLat dynamics MLP head (h_t + e_{t+1} -> h_{t+1}), multi-step rollout, and speculative decoding
  evaluate.py            # Batched code generation and sandboxed execution for test-100.jsonl and HumanEval
  launch_ablation.py     # Parallel multi-sandbox GPU launcher with live W&B streaming and paired bootstrap CIs
  tests/
    test_extract.py      # 13 unit tests covering extraction, split disjointness, mutants, repo warmup, NextLat, and CIs
  results/
    dataset_manifest.json          # Provenance, counts, and SHA-256 hashes of frozen dataset splits
    combined_ablation_summary.json # Unified 15-arm metrics, speculative decoding benchmarks, and 10,000-resample paired bootstrap CIs
    full-sft-ablation/             # Per-arm config.json, metrics.jsonl, and summary.json for base & NTP SFT
    nextlat-ablation/              # Per-arm config.json, metrics.jsonl, and summary.json for cold-start NextLat SFT
    nextlat-warmup-ablation/       # Per-arm config.json, metrics.jsonl, and summary.json for repo-warmed NextLat SFT
    spec-decode-ablation/          # Speculative decoding benchmark results on test-100.jsonl
    spec-decode-humaneval/         # Speculative decoding benchmark + functional pass@1 verification on HumanEval
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
| **`nextlat-warmup-sft-1000`** | **Repo-Warmed NextLat (1-step)** | 1,000 | 250 | `0.2690` | `23.00%` | `41.13%` | `9.18%` | `13.83%` | [`6ea757il`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/6ea757il) |
| **`spec-joint-rollout3-1000`** | **Repo-Warmed NextLat (3-step)** | 1,000 | 250 | `0.2696` | `23.70%` | `39.13%` | `9.91%` | `15.23%` | [`utoq9x6z`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/utoq9x6z) |
| **`nextlat-cpt-sft-1000`** | **Repo-Warmed + Joint CPT NextLat** | 1,000 | 250 | `0.2654` | `22.65%` | `38.17%` | `9.85%` | `14.05%` | [`60vltons`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/60vltons) |
| **`nextlat-warmup-sft-10000`** | **Repo-Warmed NextLat (1-step)** | 10,000 | 2,500 | `0.2080` | **`40.85%`** | `48.77%` | `13.17%` | `17.40%` | [`qj81ntc6`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/qj81ntc6) |

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
| **1,000 (`rollout3` vs `full-sft-1000`)** | `-1.55%` `[-6.30%, +3.30%]` | `-2.87%` `[-10.36%, +4.31%]` | `-0.98%` `[-2.71%, +0.67%]` | `+0.34%` `[-2.52%, +3.15%]` |
| **1,000 (`cpt` vs `full-sft-1000`)** | `-2.60%` `[-8.30%, +2.85%]` | `-3.83%` `[-11.93%, +4.29%]` | `-1.04%` `[-2.59%, +0.46%]` | `-0.85%` `[-3.38%, +1.70%]` |
| **10,000 (`warmup` vs `full-sft-10000`)** | **`+2.25%`** `[-2.70%, +7.30%]` | `-3.11%` `[-10.53%, +4.22%]` | `-0.46%` `[-2.44%, +1.55%]` | `-0.76%` `[-3.60%, +1.96%]` |

---

## 4. Lossless Self-Speculative Decoding with `NextLatDynamicsModel`

Because `NextLatDynamicsModel` (`~67M` parameters, $90\times$ smaller than GPT-J 6B) is trained to predict $h_{t+1} = f_\phi(h_t, e_{t+1})$ and match GPT-J's next-token logits via `lm_head`, we retain `NextLatDynamicsModel` at inference time as an EAGLE-style latent speculative drafter. Given GPT-J's last hidden state $h_t$ and greedily sampled token $\hat{y}_{t+1}$, `NextLatDynamicsModel` autoregressively rolls out $K \in \{1, 2, 3, 4\}$ draft tokens using only embedding lookups, its 2-layer MLP, and `lm_head`, which GPT-J then verifies in a single parallel forward pass.

### 4.1 Out-of-Distribution `HumanEval` (164 Tasks, `max_new_tokens=128`)

| Arm (`1,000` SFT Trajectories) | Draft $K$ | Pos-1 / Pos-2 / Pos-3 / Pos-4 Acceptance | Tokens / GPT-J Forward | Throughput (`tok/s`) | Wall-Clock Speedup | Greedy `pass@1` | W&B Run |
|---|---:|---|---:|---:|---:|---:|---|
| **`spec-humaneval-frozen-1000`** *(Frozen GPT-J)* | $K=0$ (Greedy) | — | `1.00` (`20,967` fwds) | `55.3 tok/s` | `1.00x` | `11.59%` | [`1d3cupc4`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/1d3cupc4) |
| `spec-humaneval-frozen-1000` *(Frozen GPT-J)* | $K=1$ | `49.6%` | `1.49` (`14,107` fwds) | `76.8 tok/s` | `1.39x` | `12.20%` | [`1d3cupc4`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/1d3cupc4) |
| `spec-humaneval-frozen-1000` *(Frozen GPT-J)* | $K=2$ | `43.9% / 27.9%` | `1.70` (`12,324` fwds) | `85.5 tok/s` | `1.55x` | `12.20%` | [`1d3cupc4`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/1d3cupc4) |
| `spec-humaneval-frozen-1000` *(Frozen GPT-J)* | $K=3$ | `41.2% / 25.2% / 18.2%` | `1.82` (`11,494` fwds) | `88.6 tok/s` | `1.60x` | `12.20%` | [`1d3cupc4`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/1d3cupc4) |
| `spec-humaneval-frozen-1000` *(Frozen GPT-J)* | $K=4$ | `38.9% / 23.5% / 17.3% / 11.1%` | `1.88` (`11,154` fwds) | `91.0 tok/s` | `1.64x` | `12.20%` | [`1d3cupc4`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/1d3cupc4) |
| **`spec-humaneval-rollout1-1000`** *(Joint 1-step)* | $K=0$ (Greedy) | — | `1.00` (`14,226` fwds) | `51.8 tok/s` | `1.00x` | `8.54%` | [`mt54tctb`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mt54tctb) |
| `spec-humaneval-rollout1-1000` *(Joint 1-step)* | $K=1$ | **`83.1%`** | `1.80` (`7,861` fwds) | `88.2 tok/s` | **`1.70x`** | `8.54%` | [`mt54tctb`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mt54tctb) |
| `spec-humaneval-rollout1-1000` *(Joint 1-step)* | $K=2$ | `80.3% / 30.9%` | `2.07` (`6,883` fwds) | `100.5 tok/s` | **`1.94x`** | `8.54%` | [`mt54tctb`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mt54tctb) |
| `spec-humaneval-rollout1-1000` *(Joint 1-step)* | $K=3$ | `77.8% / 32.1% / 16.5%` | `2.21` (`6,431` fwds) | `106.0 tok/s` | **`2.05x`** | `8.54%` | [`mt54tctb`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mt54tctb) |
| `spec-humaneval-rollout1-1000` *(Joint 1-step)* | $K=4$ | `78.1% / 28.7% / 12.7% / 10.0%` | `2.24` (`6,391` fwds) | `105.2 tok/s` | **`2.03x`** | `8.54%` | [`mt54tctb`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mt54tctb) |
| **`spec-humaneval-rollout3-1000`** *(Joint 3-step)* | $K=0$ (Greedy) | — | `1.00` (`14,241` fwds) | `52.1 tok/s` | `1.00x` | `8.54%` | [`51g4bv0c`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/51g4bv0c) |
| `spec-humaneval-rollout3-1000` *(Joint 3-step)* | $K=1$ | **`81.1%`** | `1.78` (`7,919` fwds) | `87.1 tok/s` | **`1.67x`** | `8.54%` | [`51g4bv0c`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/51g4bv0c) |
| `spec-humaneval-rollout3-1000` *(Joint 3-step)* | $K=2$ | **`77.5% / 59.4%`** | `2.31` (`6,052` fwds) | `111.8 tok/s` | **`2.15x`** | `9.15%` | [`51g4bv0c`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/51g4bv0c) |
| `spec-humaneval-rollout3-1000` *(Joint 3-step)* | $K=3$ | **`76.7% / 57.3% / 43.0%`** | `2.69` (`5,326` fwds) | `128.1 tok/s` | **`2.46x`** | `9.15%` | [`51g4bv0c`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/51g4bv0c) |
| `spec-humaneval-rollout3-1000` *(Joint 3-step)* | $K=4$ | **`73.0% / 53.2% / 39.7% / 28.7%`** | **`2.84`** (`4,953` fwds) | **`133.2 tok/s`** | **`2.56x`** | `9.15%` | [`51g4bv0c`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/51g4bv0c) |

#### Scaling to `10,000` SFT Trajectories on `HumanEval` (164 Tasks, `max_new_tokens=128`)

| Arm (`10,000` SFT Trajectories) | Draft $K$ | Pos-1 / Pos-2 / Pos-3 / Pos-4 Acceptance | Tokens / GPT-J Forward | Throughput (`tok/s`) | Wall-Clock Speedup | Greedy `pass@1` | W&B Run |
|---|---:|---|---:|---:|---:|---:|---|
| **`spec-humaneval-rollout1-10000`** *(Joint 1-step)* | $K=0$ (Greedy) | — | `1.00` (`12,336` fwds) | `56.3 tok/s` | `1.00x` | `13.41%` | [`mw1x12mv`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mw1x12mv) |
| `spec-humaneval-rollout1-10000` *(Joint 1-step)* | $K=1$ | **`88.4%`** | `1.86` (`6,746` fwds) | `94.1 tok/s` | **`1.67x`** | `13.41%` | [`mw1x12mv`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mw1x12mv) |
| `spec-humaneval-rollout1-10000` *(Joint 1-step)* | $K=2$ | `86.2% / 42.4%` | `2.24` (`5,573` fwds) | `108.6 tok/s` | **`1.93x`** | `13.41%` | [`mw1x12mv`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mw1x12mv) |
| `spec-humaneval-rollout1-10000` *(Joint 1-step)* | $K=3$ | `85.3% / 41.2% / 22.9%` | `2.43` (`5,104` fwds) | `114.8 tok/s` | **`2.04x`** | `13.41%` | [`mw1x12mv`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mw1x12mv) |
| `spec-humaneval-rollout1-10000` *(Joint 1-step)* | $K=4$ | `84.2% / 38.3% / 19.8% / 14.8%` | `2.50` (`4,938` fwds) | `116.2 tok/s` | **`2.06x`** | `13.41%` | [`mw1x12mv`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/mw1x12mv) |
| **`spec-humaneval-rollout3-10000`** *(Joint 3-step)* | $K=0$ (Greedy) | — | `1.00` (`12,489` fwds) | `52.0 tok/s` | `1.00x` | `13.41%` | [`5g754cmg`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/5g754cmg) |
| `spec-humaneval-rollout3-10000` *(Joint 3-step)* | $K=1$ | **`87.3%`** | `1.85` (`6,744` fwds) | `88.2 tok/s` | **`1.70x`** | `14.02%` | [`5g754cmg`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/5g754cmg) |
| `spec-humaneval-rollout3-10000` *(Joint 3-step)* | $K=2$ | **`86.3% / 72.1%`** | `2.52` (`4,910` fwds) | `116.5 tok/s` | **`2.24x`** | `14.02%` | [`5g754cmg`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/5g754cmg) |
| `spec-humaneval-rollout3-10000` *(Joint 3-step)* | $K=3$ | **`84.9% / 70.8% / 57.4%`** | `3.03` (`4,154` fwds) | `138.4 tok/s` | **`2.66x`** | `13.41%` | [`5g754cmg`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/5g754cmg) |
| `spec-humaneval-rollout3-10000` *(Joint 3-step)* | $K=4$ | **`83.9% / 68.4% / 55.0% / 40.8%`** | **`3.34`** (`3,707` fwds) | **`149.7 tok/s`** | **`2.88x`** | `14.63%` | [`5g754cmg`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/5g754cmg) |

### 4.2 In-Domain `test-100` (100 Tasks, `max_new_tokens=128`)

| Arm (`1,000` SFT Trajectories) | Draft $K$ | Pos-1 / Pos-2 / Pos-3 / Pos-4 Acceptance | Tokens / GPT-J Forward | Throughput (`tok/s`) | Wall-Clock Speedup | W&B Run |
|---|---:|---|---:|---:|---:|---|
| **`spec-frozen-drafter-1000`** *(Frozen GPT-J)* | $K=0$ (Greedy) | — | `1.00` (`10,557` fwds) | `55.3 tok/s` | `1.00x` | [`gtotav6g`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/gtotav6g) |
| `spec-frozen-drafter-1000` *(Frozen GPT-J)* | $K=3$ | `36.4% / 22.0% / 12.1%` | `1.54` (`6,858` fwds) | `76.6 tok/s` | `1.39x` | [`gtotav6g`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/gtotav6g) |
| **`spec-joint-rollout1-1000`** *(Joint 1-step)* | $K=0$ (Greedy) | — | `1.00` (`7,095` fwds) | `52.3 tok/s` | `1.00x` | [`epbmsoox`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/epbmsoox) |
| `spec-joint-rollout1-1000` *(Joint 1-step)* | $K=1$ | **`80.9%`** | `1.78` (`3,991` fwds) | `87.1 tok/s` | **`1.67x`** | [`epbmsoox`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/epbmsoox) |
| `spec-joint-rollout1-1000` *(Joint 1-step)* | $K=3$ | `75.7% / 27.9% / 14.2%` | `2.12` (`3,354` fwds) | `101.0 tok/s` | **`1.93x`** | [`epbmsoox`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/epbmsoox) |
| **`spec-joint-rollout3-1000`** *(Joint 3-step)* | $K=0$ (Greedy) | — | `1.00` (`6,850` fwds) | `48.0 tok/s` | `1.00x` | [`utoq9x6z`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/utoq9x6z) |
| `spec-joint-rollout3-1000` *(Joint 3-step)* | $K=2$ | **`73.6% / 52.2%`** | `2.20` (`3,131` fwds) | `98.8 tok/s` | **`2.06x`** | [`utoq9x6z`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/utoq9x6z) |
| `spec-joint-rollout3-1000` *(Joint 3-step)* | $K=3$ | **`71.9% / 50.6% / 37.3%`** | `2.50` (`2,735` fwds) | `111.2 tok/s` | **`2.32x`** | [`utoq9x6z`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/utoq9x6z) |
| `spec-joint-rollout3-1000` *(Joint 3-step)* | $K=4$ | **`69.7% / 46.0% / 34.2% / 23.1%`** | **`2.62`** (`2,595` fwds) | **`114.6 tok/s`** | **`2.39x`** | [`utoq9x6z`](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft/runs/utoq9x6z) |

---

## 5. Key Takeaways

1. **Why Cold-Start NextLat Struggled in Post-Training:**
   - Attaching a randomly initialized `NextLatDynamicsModel` (`~67M` parameters) directly to a pretrained 6B model (`GPT-J`) at step 0 with $\lambda_{\text{MSE}}=1.0, \lambda_{\text{KL}}=1.0$ creates a massive initial KL loss (`~9.72` vs. `0.32` NTP loss) and a `grad_norm` spike of `~51`, scrambling the pretrained backbone during the first ~100–250 steps (`val-100` NLL jumped from `0.3211` to `0.5933` at 100 examples).
2. **Frozen-Backbone Repo Warmup + Zero-Init Completely Fixes NextLat:**
   - Initializing `NextLatDynamicsModel`'s final residual projection to zero (`zero_init_output=True`), warming up only the dynamics head for 250 steps (`1,000` decontaminated Python repo functions under `torch.no_grad()` on `GPT-J`), and scaling the auxiliary loss weights ($\lambda_{\text{MSE}}=0.2, \lambda_{\text{KL}}=0.1$) boosted NextLat by **`+4.90%` to `+11.15%` `pass@1`** across all dataset sizes:
     - At **100 examples**, `nextlat-warmup-sft-100` **beats `full-sft-100` on `HumanEval pass@1`** (`7.01%` vs `5.27%`, `+1.74%`, 95% CI `[+0.09%, +3.78%]`) and `test-100 pass@10` (`22.47%` vs `20.84%`).
     - At **1,000 examples**, `test-100 pass@1` nearly doubles over cold-start (`12.35% -> 23.00%–23.70%`, **`+10.65% to +11.35%`**), matching `full-sft-1000` within statistical noise (all 95% CIs overlap zero).
     - At **10,000 examples**, `nextlat-warmup-sft-10000` achieves **`40.85%` `pass@1` on `test-100`**—**the highest `pass@1` score of any model in the entire study**, beating cold-start `nextlat-sft-10000` (`29.70%`) by **`+11.15%`** and surpassing `full-sft-10000` (`38.60%`) by **`+2.25%`** (`+23.10%` over the `17.75%` base GPT-J model).
3. **Free `2.05x–2.88x` Lossless Speculative Decoding via Native Hugging Face `GPTJNextLatForCausalLM`:**
   - Training `NextLatDynamicsModel` jointly during SFT co-adapts GPT-J's hidden states $h_t$ and the 67M drafter head: on `HumanEval`, step-1 draft acceptance jumps from **`49.6%`** (frozen GPT-J drafter, max `1.64x` speedup) to **`81.1%–83.1%`** at 1,000 SFT trajectories and **`87.3%–88.4%`** at 10,000 SFT trajectories.
   - Adding a 3-step latent rollout (`--rollout-steps 3`) during training nearly doubles multi-step acceptance at positions 2–4:
     - At **1,000 SFT trajectories** (`spec-humaneval-rollout3-1000`), acceptance reaches `73.0% / 53.2% / 39.7% / 28.7%` at $K=4$, yielding **`2.84` tokens/forward** and a **`2.56x` wall-clock speedup (`52.1 -> 133.2 tok/s`) on `HumanEval`** (`2.39x` on `test-100`).
     - At **10,000 SFT trajectories** (`spec-humaneval-rollout3-10000`), acceptance climbs further to **`83.9% / 68.4% / 55.0% / 40.8%`** at $K=4$, cutting 6B target forward passes by **`3.37x`** (`12,489 -> 3,707` fwds, **`3.34` tokens/forward**) and achieving a **`2.88x` wall-clock speedup (`52.0 -> 149.7 tok/s`) on `HumanEval`** while improving functional `pass@1` from `8.54%` (1k) to **`13.41%–14.63%`** (10k).
