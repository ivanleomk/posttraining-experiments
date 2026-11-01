"""HumanEval on GPT-J, optionally loading a LoRA adapter. Run on an isolated GPU sandbox.
Install: pip install torch==2.8.0 transformers==4.56.2 peft==0.17.1 pydantic==2.12.3
Install official evaluator:
  pip install https://github.com/openai/human-eval/archive/b9453e8f1865df0a0057a1383767caa400acfbcf.zip
Base: python evaluate.py --output /tmp/base-eval --execute-in-sandbox
LoRA: python evaluate.py --adapter /tmp/lora-100/adapter --output /tmp/lora-eval --execute-in-sandbox
Executes generated Python: use a disposable sandbox without credentials or host mounts.
Defaults match our earlier run: 20 samples/task, temperature .2, top_p .95,
top_k 0, max 512 new tokens, batch 10, seed 42; reports pass@1 and pass@10.
This is a reduced-sample protocol, not a full replication of the Codex paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, ValidationError

MODEL_ID = "EleutherAI/gpt-j-6b"
MODEL_REVISION = "47e169305d2e8376be1d31e765533382721b2cc1"
STOPS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output: Path
    adapter: DirectoryPath | None = None
    samples: int = Field(default=20, ge=10)
    batch_size: int = Field(default=10, ge=1)
    max_new_tokens: int = Field(default=512, ge=1)
    seed: int = 42
    execute_in_sandbox: Literal[True]


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute-in-sandbox", action="store_true")
    try:
        return EvalConfig.model_validate(vars(p.parse_args(argv)))
    except ValidationError as error:
        p.error(str(error))


def prepare_benchmark(output: Path) -> tuple[dict[str, Any], Path, dict[str, float]]:
    from human_eval.data import read_problems
    from human_eval.evaluation import evaluate_functional_correctness

    problems = read_problems()
    problem_file = output / "problems.jsonl"
    problem_file.write_text(
        "".join(json.dumps(dict(v, task_id=k)) + "\n" for k, v in problems.items())
    )
    canonical = output / "canonical.jsonl"
    canonical.write_text(
        "".join(
            json.dumps(dict(task_id=k, completion=v["canonical_solution"])) + "\n"
            for k, v in problems.items()
        )
    )
    control = evaluate_functional_correctness(
        str(canonical), k=[1], n_workers=4, timeout=3, problem_file=str(problem_file)
    )
    if control["pass@1"] != 1:
        raise RuntimeError(f"Canonical control failed: {control}")
    return problems, problem_file, control


def load_evaluation_model(config: EvalConfig) -> tuple[Any, Any, dict[str, str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_hashes = {}
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.float16,
            use_safetensors=False,
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )
    if config.adapter:
        from peft import PeftModel

        adapter_config = json.loads(
            (config.adapter / "adapter_config.json").read_text()
        )
        if adapter_config["base_model_name_or_path"] != MODEL_ID:
            raise ValueError("Adapter base model differs from GPT-J")
        adapter_hashes = {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in config.adapter.iterdir()
            if f.is_file()
        }
        model = PeftModel.from_pretrained(model, config.adapter).eval()
    return model, tok, adapter_hashes


def generate_samples(
    model: Any, tok: Any, problems: dict[str, Any], config: EvalConfig
) -> tuple[Path, int]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, set_seed

    class Stop(StoppingCriteria):
        def __init__(self, start):
            self.start = start

        def __call__(self, input_ids, scores, **kwargs):
            texts = tok.batch_decode(
                input_ids[:, self.start :], skip_special_tokens=True
            )
            return torch.tensor(
                [any(s in text for s in STOPS) for text in texts],
                device=input_ids.device,
            )

    cap_count = 0
    sample_path = config.output / "samples.jsonl"
    with (
        sample_path.open("w") as samples,
        (config.output / "raw.jsonl").open("w") as raw,
    ):
        for index, (task_id, problem) in enumerate(problems.items()):
            inputs = tok(problem["prompt"], return_tensors="pt").to("cuda")
            prompt_len = inputs["input_ids"].shape[1]
            if (
                prompt_len + config.max_new_tokens
                > model.config.max_position_embeddings
            ):
                raise ValueError(f"Context overflow: {task_id}")
            for offset in range(0, config.samples, config.batch_size):
                set_seed(config.seed + index * 1000 + offset)
                n = min(config.batch_size, config.samples - offset)
                with torch.inference_mode():
                    output = model.generate(
                        **{k: v.repeat(n, 1) for k, v in inputs.items()},
                        do_sample=True,
                        temperature=0.2,
                        top_p=0.95,
                        top_k=0,
                        max_new_tokens=config.max_new_tokens,
                        pad_token_id=tok.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([Stop(prompt_len)]),
                    )
                for seq in output[:, prompt_len:]:
                    text = tok.decode(seq, skip_special_tokens=True)
                    stops = [text.find(s) for s in STOPS if s in text]
                    hit_cap = (
                        len(seq) == config.max_new_tokens
                        and not stops
                        and tok.eos_token_id not in seq.tolist()
                    )
                    cap_count += int(hit_cap)
                    raw.write(
                        json.dumps(
                            dict(
                                task_id=task_id, completion=text, hit_output_cap=hit_cap
                            )
                        )
                        + "\n"
                    )
                    samples.write(
                        json.dumps(
                            dict(
                                task_id=task_id,
                                completion=text[: min(stops)] if stops else text,
                            )
                        )
                        + "\n"
                    )
                raw.flush()
                samples.flush()
            print(
                json.dumps(dict(tasks_completed=index + 1, tasks_total=len(problems))),
                flush=True,
            )
    return sample_path, cap_count


def score_samples(sample_path: Path, problem_file: Path) -> dict[str, float]:
    from human_eval.evaluation import evaluate_functional_correctness

    return evaluate_functional_correctness(
        str(sample_path),
        k=[1, 10],
        n_workers=4,
        timeout=3,
        problem_file=str(problem_file),
    )


def evaluate_model(config: EvalConfig) -> dict[str, Any]:
    """Validate the harness, generate completions, then score with HumanEval."""
    import torch
    import transformers

    if config.output.exists():
        raise FileExistsError(f"{config.output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    config.output.mkdir(parents=True)
    problems, problem_file, control = prepare_benchmark(config.output)
    model, tokenizer, adapter_hashes = load_evaluation_model(config)
    metadata = dict(
        config.model_dump(mode="json"),
        model=MODEL_ID,
        revision=MODEL_REVISION,
        temperature=0.2,
        top_p=0.95,
        top_k=0,
        stops=STOPS,
        torch=torch.__version__,
        transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0),
        adapter_files_sha256=adapter_hashes,
        dataset_sha256=hashlib.sha256(
            json.dumps(problems, sort_keys=True).encode()
        ).hexdigest(),
    )
    (config.output / "config.json").write_text(json.dumps(metadata, indent=2))
    start = time.monotonic()
    samples, cap_count = generate_samples(model, tokenizer, problems, config)
    scores = score_samples(samples, problem_file)
    summary = dict(
        status="complete",
        config=metadata,
        scores=scores,
        canonical_control=control,
        output_cap_hits=cap_count,
        elapsed_seconds=time.monotonic() - start,
    )
    (config.output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    config = parse_args()
    summary = evaluate_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
