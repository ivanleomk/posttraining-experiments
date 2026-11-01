# GPT-J code SFT: dataset size and training objective

We want to measure whether a small amount of code instruction tuning improves GPT-J's coding ability, how that improvement changes with dataset size, and whether a NextLat-style auxiliary objective improves on standard next-token prediction (NTP).

We will train on nested BigCode subsets, evaluate on 100 held-out BigCode problems with independently generated tests, and also report HumanEval. Every training arm starts from the same original GPT-J checkpoint. This is a LoRA experiment; full-parameter fine-tuning is a separate ablation.

## What exists today

- `generate_splits.py`: downloads pinned inputs and generates nested 100/1,000/10,000 training pools plus a separate 100-row holdout. It performs static filtering and structural overlap checks.
- `train_lora.py`: answer-only NTP LoRA training on GPT-J.
- `evaluate.py`: HumanEval generation and execution, optionally loading a LoRA adapter.
- `tests/test_extract.py`: tests for the static code extractor.

The existing `test-100` contains **reference solutions, not executable tests**. Gemini test generation, a separate validation split, custom-benchmark evaluation, and the NextLat-style objective still need implementation. This README is the plan, not a report of completed runs.

## 1. Freeze the data

Use [BigCode Self-OSS-Instruct SC2 execution-filtered 50K](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k), revision `356bb069eee815daa6e23e9a282eeefe1490ad44`. It contains 50,661 synthetic instruction/response examples produced for StarCoder2 self-alignment, grounded in open-source Python functions. Preserve ODC-By attribution and per-example provenance.

Use `EleutherAI/gpt-j-6b`, revision `47e169305d2e8376be1d31e765533382721b2cc1`, for both model and tokenizer.

| Pool | Size | Purpose |
|---|---:|---|
| Training | 10,000 | Frozen master pool; take exact prefixes of 100 and 1,000 |
| Validation | 100 | Reference-completion loss and development decisions |
| Custom evaluation | 100 | Frozen executable correctness benchmark |
| HumanEval | 164 | Separate standard executable benchmark |

Keep the current training pool and `test-100` membership unchanged. Add 100 validation examples from unused source families, excluding all families present in either existing pool. Keep source seed, normalized instruction and canonical solution-body families disjoint across all three BigCode pools. If the eligible pool cannot satisfy this, stop and explicitly version a new split rather than silently changing membership.

The existing builder rejects unsupported snippets, unresolved global references and sequences longer than 2,048 tokens; it deduplicates normalized solution bodies and screens normalized HumanEval overlaps. These checks are not semantic decontamination or proof of reference correctness. Do not execute dataset code on the host.

Keep model-facing data small:

```json
{"prompt": "def example(...):\n    'Task description'\n", "completion": "    # reference solution\n"}
```

Store IDs, source revision, family membership, filtering decisions and hashes in an accompanying manifest/provenance file. Keep a stable ID linking each example to that record. The current builder embeds these fields in each row; separating them is a planned format cleanup.

**Done when:** counts, nesting, family disjointness, token lengths and repeat-build hashes are checked, and all split IDs are frozen.

## 2. Build executable tests for the held-out 100

Use Gemini to generate tests for the existing 100 held-out BigCode problems. This creates a small custom benchmark; it does not produce training examples.

1. Supply the task instruction, exact function signature and a fixed test-generation prompt. Initially hide the reference solution and all model predictions.
2. Request ordinary Python assertions in a `check(candidate)` function. Cover typical inputs, specified boundaries and meaningful edge cases. Do not invent behavior for unspecified inputs.
3. Record the exact Gemini model ID, generation settings, prompt version, raw response, request outcome and per-problem attempt count. Set the generation budget before starting: at most three attempts per problem, then manual review.
4. Parse the response, check the callable contract and run it against the reference solution in a disposable execution sandbox with time/resource limits.
5. Review failures against the instruction. A failing test can indicate a bad test, a bad reference, or an ambiguous task. Do not automatically rewrite tests to agree with the reference. Record any repair and its justification.
6. Check test strength using deliberately incorrect implementations: constant returns, off-by-one boundaries, reversed conditions or other task-relevant errors. Record which non-equivalent mutants each suite rejects; equivalent mutants do not count as failures of test quality.
7. Review every task/test contract. Include at least one normal case and one specified boundary case when applicable, and require a documented, task-specific incorrect implementation to be caught. Hold ambiguous or weakly tested tasks for review.
8. Freeze the accepted tasks, reference solutions, test suites and hashes before evaluating trained adapters. Never alter tests in response to a model's score.

If a problem cannot be made unambiguous, retain its exclusion record and report the final denominator. Do not replace it opportunistically or claim a 100-task score when fewer tasks were accepted. Choose any replacement policy before running models.

The evaluation record should contain:

```json
{
  "task_id": "bigcode/<source_id>",
  "prompt": "def example(...):\n    'Task description'\n",
  "entry_point": "example",
  "canonical_solution": "    # reference body\n",
  "test": "def check(candidate):\n    # assertions\n"
}
```

Only `prompt` goes to the model. References and tests are used by the evaluator. Save generated predictions separately as `{task_id, completion}`. Passing a generated test suite is evidence of correctness under those tests, not proof of complete correctness.

**Done when:** all accepted references pass, deliberately incorrect implementations are caught, exclusions and repairs are audited, and the benchmark is frozen.

## 3. Establish the baseline

Extend `evaluate.py` to accept the custom benchmark as well as HumanEval, using the same generation and scoring functions for base and adapter runs.

- Run the original GPT-J checkpoint on both benchmarks before training.
- Verify every canonical reference passes its benchmark tests.
- Generate 20 completions per task, in batches of 10, with seed 42, temperature 0.2, top-p 0.95, top-k 0 and a 512-new-token cap.
- Use the existing stop sequences: `\nclass`, `\ndef`, `\n#`, `\nif`, `\nprint`. Check compatibility with the custom tasks before freezing the protocol.
- Compute pass@1 and pass@10 with the standard estimator over all 20 samples; do not select the best sample or report “any of 20” as pass@10.
- Preserve raw/completion text, per-sample outcomes, runtime failures, output-cap hits, task IDs, package versions, GPU type and model/adapter hashes.

Run generated code only in disposable sandboxes without credentials or host mounts. Code-level filtering and the evaluator's timeout are not substitutes for isolation.

**Done when:** both benchmark controls pass and the base-model artifacts and scores are saved.

## 4. Run the training ablation

| Training examples | NTP LoRA | NTP + NextLat-style LoRA |
|---|---|---|
| 100 | Run | Run |
| 1,000 | Run | Run |
| 10,000 | Run | Run |

First establish the three NTP runs. Then implement and verify the auxiliary objective before running its three corresponding arms.

Shared starting recipe:

- Fresh GPT-J base checkpoint for every run; never continue from the previous subset's adapter.
- LoRA rank 8, alpha 16, dropout 0, targeting `q_proj` and `v_proj`.
- One epoch, microbatch size 1, gradient accumulation 4.
- AdamW, learning rate `2e-4`, weight decay `0.01`, cosine schedule, one warmup optimizer step and gradient clipping at 1.0.
- BF16 on supported GPUs. Keep dtype and GPU type consistent within matched comparisons; record any deviations.
- Same concatenated prompt/completion tokenization, prompt labels masked to `-100`, completion and EOS supervised. No truncation.
- Token-weight accumulated gradients, including the final partial group.

One epoch is the primary dataset-size comparison, so larger datasets also receive more tokens and optimizer steps. Report this explicitly; it does not isolate diversity from compute. A matched-token-budget experiment can follow separately.

Use the final checkpoint after the fixed epoch for the primary result. Monitor validation loss for diagnostics; do not choose checkpoints or hyperparameters using either executable evaluation benchmark. If we tune a recipe on validation, freeze it before the full comparison and document that tuning.

### Implementing the auxiliary objective

Use [NextLat](https://github.com/JaydenTeoh/NextLat) as the reference method, and pin the paper/code revision before implementation. Our GPT-J LoRA adaptation is an experiment, not an existing validated result.

Write down the exact prediction head, hidden-state layers, future offsets, detached targets, loss terms and coefficients before training. Retain NTP and implement the selected NextLat components explicitly; adjacent-state MSE alone should not be labeled a faithful implementation. Auxiliary targets must stay inside the answer and the same example. The head can be discarded for standard autoregressive evaluation.

Verify that gradients reach the intended trainable parameters, detached targets receive no gradient through the target branch, prompt/padding/boundary masks work, and disabling the auxiliary loss recovers the NTP path. Save head parameters and objective configuration with training artifacts.

Choose any auxiliary-loss coefficients using validation only. Keep data, example order, base checkpoint, adapter capacity and generation settings matched to the NTP arm. Record extra parameters, tokens, GPU time and memory introduced by the objective.

**Done when:** the six run artifacts are complete, adapter reloads work, and training metadata identifies the exact data and objective used.

## 5. Evaluate and interpret

Evaluate each final adapter on both frozen benchmarks. For every result, report:

- Pass@1 and pass@10, plus the change from the base model and the paired NTP arm.
- Task-level paired bootstrap confidence intervals for score differences, resampling the same task IDs for both models.
- Validation answer-token loss, processed tokens, optimizer steps, training time and peak GPU memory.
- Invalid completions, execution errors and output-cap rate.

The first pass uses one training seed (42): six adapters plus one base baseline, each evaluated on two benchmarks. Bootstrap intervals describe task uncertainty conditional on those trained models and sampled completions; they do not capture training-seed variance. Before treating an objective advantage as robust, repeat the relevant matched comparison with seeds 43 and 44 and report all three.

Keep results from the custom test set and HumanEval separate. Report small or uncertain changes honestly. Our previous 93-example pilot is a pipeline check, not the 100-example arm of this frozen ladder.

## Implementation order

- [ ] Add the separate validation pool without changing existing training/evaluation membership.
- [ ] Simplify training rows and preserve provenance separately.
- [ ] Add Gemini test generation with versioned prompts, bounded retries and raw-response capture.
- [ ] Add sandbox validation, test-strength checks and a reviewed acceptance/exclusion manifest.
- [ ] Freeze the custom benchmark and extend `evaluate.py` to load it.
- [ ] Save GPT-J base results on both benchmarks.
- [ ] Run and evaluate NTP LoRA at 100, 1,000 and 10,000 examples.
- [ ] Specify, implement and verify the NextLat-style objective.
- [ ] Run and evaluate its three matched dataset-size arms.
- [ ] Produce the comparison table and paired intervals; repeat relevant comparisons across seeds.

Keep the three existing scripts as the entry points: parse arguments into a typed Pydantic configuration, then call the workflow function. Add focused helpers and tests for the new behavior as needed. No training or Gemini calls are initiated by this plan.
