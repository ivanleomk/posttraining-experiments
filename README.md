# Post-Training Experiments

Runnable post-training experiments and structured artifacts for the [posttraining](https://github.com/ivanleomk/posttraining) notebook.

## Experiments

- [**GPT-J 6B Code SFT: Dataset Scaling & NextLat vs. Standard NTP**](code-sft-data/README.md)
  - **Model:** `EleutherAI/gpt-j-6b` (`47e169305d2e8376be1d31e765533382721b2cc1`)
  - **Dataset:** Nested 100 / 1,000 / 10,000 Python SFT splits + 100 verified validation tasks (`val-100.jsonl`) + 100 verified test tasks (`test-100.jsonl`) from `bigcode/self-oss-instruct-sc2-exec-filter-50k` (`356bb069eee815daa6e23e9a282eeefe1490ad44`), with single-shot `models/gemini-3.8-flash` executable unit test suites verified against reference solutions and AST/generic mutants.
  - **Ablations:** 7 full-parameter arms (`base`, `full-sft-{100,1000,10000}`, `nextlat-sft-{100,1000,10000}`) evaluated on `val-100` answer-token NLL, in-domain `test-100` (`pass@1`, `pass@10`), and out-of-distribution `HumanEval` (`pass@1`, `pass@10`) with 10,000-resample paired bootstrap 95% confidence intervals.
  - **Artifacts:** [`code-sft-data/results/`](code-sft-data/results/) and [Weights & Biases (`ivanleomk-experiments/gptj-code-sft`)](https://forge.coreweave.com/wandb/ivanleomk-experiments/gptj-code-sft).
