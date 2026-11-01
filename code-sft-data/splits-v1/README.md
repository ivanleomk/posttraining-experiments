# Code-SFT dataset-size ladder v1

Pinned source: [BigCode self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k), revision `356bb069eee815daa6e23e9a282eeefe1490ad44`, ODC-By. Preserve source attribution on redistribution.

| File | Examples | Total tokens incl. prompt/EOS | Supervised answer tokens incl. EOS |
|---|---:|---:|---:|
| train-100.jsonl | 100 | 25,426 | 9,724 |
| train-1000.jsonl | 1,000 | 247,284 | 95,509 |
| train-10000.jsonl | 10,000 | 2,461,143 | 975,820 |
| test-100.jsonl | 100 | 25,134 | 9,413 |

The smaller training sets are exact prefixes of `train-10000.jsonl`. All use the same shuffled order (seed 42); the fixed test set is outside the training pool. This test set is an in-domain reference-loss holdout, **not HumanEval** and not an executable benchmark. Do not use it for training. If it is used to tune hyperparameters, call it development data rather than a final unbiased test set.

Start each dataset-size experiment from the **same original GPT-J checkpoint**, not the previously trained 93-example adapter. That older pilot used a different data selection and is not the 100-example arm of this ladder.

## Processing

All 50,661 upstream rows were inspected. We extracted single-function Python code from the response, kept the original instruction as a docstring, removed explanations, and normalized code with AST. Imports, multi-function snippets, selected I/O/dynamic-execution calls, unresolved global references, decorators, and examples exceeding GPT-J's 2,048-token context were filtered out. Source example 533341 is excluded due to known contract ambiguity.

Normalized solution bodies are deduplicated, including simple local-variable renaming. Related rows are grouped by upstream seed hash, normalized instruction hash, and canonical solution-body hash **before** deduplication; entire holdout families are kept out of training. The source ID and row index remain in every record. This family grouping is conservative structural processing, not a semantic-equivalence proof.

63 candidates matched normalized HumanEval task/solution fingerprints and were excluded. HumanEval is used only for overlap screening, never to synthesize or select training answers. Screening does not guarantee absence of paraphrases or other semantic contamination.

24,977 unique candidates survived. A deterministic shuffle chooses the 100 test rows first and reserves their entire families; the first 10,000 remaining rows form the training pool. Full selection/rejection statistics and file checksums are in `manifest.json`; per-row filter rejections are in `rejected.jsonl`.

## Training format

Each JSONL record includes `prompt`, `completion`, source provenance, family hashes, and token counts. NTP and NextLat arms should use the **same rows and tokenization**. Tokenize the concatenation with GPT-J's pinned tokenizer, mask prompt labels with -100 using offsets, and supervise completion/EOS. Do not silently truncate rows or cross example boundaries with latent targets.

These are **upstream execution-filtered, statically curated candidates**. We have not individually rerun tests or reviewed the contracts of all 10,100 examples. Static parsing does not execute code or certify correctness. The previous three reviewed preview cases do not imply that this whole pool was manually validated.

## Rebuild

From the parent `code-sft-data` directory, with its Python 3.11 environment and pinned dependencies:

```bash
.venv/bin/python build_splits.py --output splits-v2
.venv/bin/python -m unittest discover -s tests -v
```

Requires the pinned `source.parquet` and `humaneval-reference.jsonl.gz` already present in the parent directory. Both input hashes are verified. The GPT-J tokenizer is downloaded at a pinned revision. Existing output directories are refused, so rebuilding never overwrites a frozen pool. No GPU, training, or dataset-code execution is involved.
