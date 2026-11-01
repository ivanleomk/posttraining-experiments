# GPT-J code-SFT dataset-size ablation

Nested training pools of 100, 1,000 and 10,000 examples, plus a separate 100-example in-domain holdout. This prepares controlled dataset-size comparisons before comparing NTP and NextLat-style objectives. No training is launched.

See [split documentation](splits-v1/README.md) for counts, provenance, filtering, training format and limitations. Generated JSONL files and checksums are committed for review.

## Reproduce

Use Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -fL https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k/resolve/356bb069eee815daa6e23e9a282eeefe1490ad44/data/train-00000-of-00001.parquet -o source.parquet
curl -fL https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz -o humaneval-reference.jsonl.gz
.venv/bin/python build_splits.py --output splits-v2
.venv/bin/python -m unittest discover -s tests -v
```

Both input hashes are checked. Source downloads are excluded from Git. `build_preview.py` supplies shared extraction logic and pinned revision constants; run `build_splits.py` for this experiment.

Source: [BigCode self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k), revision `356bb069eee815daa6e23e9a282eeefe1490ad44`, ODC-By. Preserve attribution and row provenance when redistributing. HumanEval is used only for overlap screening. We have not individually execution-verified these candidates; structural screening does not establish semantic decontamination.
