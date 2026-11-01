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

from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    ValidationError,
)

MODEL_ID = "EleutherAI/gpt-j-6b"
MODEL_REVISION = "47e169305d2e8376be1d31e765533382721b2cc1"
STOPS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]


def truncate_at_stops(text: str, stops: list[str] = STOPS) -> str:
    """Truncate generated completion at the first occurrence of any stop string."""
    indices = [text.find(s) for s in stops if s in text]
    return text[: min(indices)] if indices else text


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output: Path
    adapter: DirectoryPath | None = None
    benchmark: FilePath | None = None
    samples: int = Field(default=20, ge=10)
    batch_size: int = Field(default=10, ge=1)
    max_new_tokens: int = Field(default=512, ge=1)
    seed: int = 42
    execute_in_sandbox: Literal[True]


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--benchmark", type=Path)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute-in-sandbox", action="store_true")
    try:
        return EvalConfig.model_validate(vars(p.parse_args(argv)))
    except ValidationError as error:
        p.error(str(error))


def prepare_benchmark(
    output: Path, benchmark: Path | None = None
) -> tuple[dict[str, Any], Path, dict[str, float]]:
    from human_eval.data import read_problems
    from human_eval.evaluation import evaluate_functional_correctness

    if benchmark is None:
        problems = read_problems()
    else:
        rows = [
            json.loads(line)
            for line in benchmark.read_text().splitlines()
            if line.strip()
        ]
        problems = {row["task_id"]: row for row in rows}
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
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if config.adapter and (config.adapter / "config.json").exists():
        adapter_hashes = {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in config.adapter.iterdir()
            if f.is_file()
        }
        model = (
            AutoModelForCausalLM.from_pretrained(
                config.adapter,
                torch_dtype=dtype,
                trust_remote_code=False,
            )
            .to("cuda")
            .eval()
        )
        return model, tok, adapter_hashes

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


def run_benchmark_eval(
    model: Any,
    tokenizer: Any,
    config: EvalConfig,
    adapter_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    import torch
    import transformers

    config.output.mkdir(parents=True, exist_ok=True)
    problems, problem_file, control = prepare_benchmark(
        config.output, config.benchmark
    )
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
        adapter_files_sha256=adapter_hashes or {},
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


def run_speculative_decoding_benchmark(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 128,
    draft_ks: tuple[int, ...] = (1, 2, 3, 4),
    task_ids: list[str] | None = None,
    problem_file: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Benchmark standard model.generate() (K=0 greedy vs K=1..4 NextLat speculative decoding)."""
    import torch
    from human_eval.evaluation import evaluate_functional_correctness

    was_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    model.eval()
    eos_id = tokenizer.eos_token_id
    device = model.device

    if prompts:
        warm = torch.tensor(
            [tokenizer(prompts[0], add_special_tokens=False)["input_ids"]],
            device=device,
        )
        model.generate(input_ids=warm, max_new_tokens=8, do_sample=False)
        model.generate(
            input_ids=warm, max_new_tokens=8, do_sample=False, num_assistant_tokens=2
        )

    baseline_tokens: list[list[int]] = []
    base_total_tokens = 0
    base_total_sec = 0.0

    for prompt in prompts:
        ids = torch.tensor(
            [tokenizer(prompt, add_special_tokens=False)["input_ids"]], device=device
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        base_total_sec += time.perf_counter() - t0
        toks = out[0, ids.size(1) :].tolist()
        baseline_tokens.append(toks)
        base_total_tokens += len(toks)

    base_tps = base_total_tokens / max(base_total_sec, 1e-9)
    greedy_summary: dict[str, Any] = {
        "total_tokens": base_total_tokens,
        "target_forwards": base_total_tokens,
        "elapsed_seconds": base_total_sec,
        "tokens_per_sec": base_tps,
        "tokens_per_target_forward": 1.0,
    }

    if task_ids is not None and problem_file is not None and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        greedy_path = output_dir / "greedy_samples.jsonl"
        with greedy_path.open("w") as f:
            for tid, toks in zip(task_ids, baseline_tokens):
                raw = tokenizer.decode(toks, skip_special_tokens=True)
                f.write(
                    json.dumps({"task_id": tid, "completion": truncate_at_stops(raw)})
                    + "\n"
                )
        greedy_summary["pass@1"] = float(
            evaluate_functional_correctness(
                str(greedy_path),
                k=[1],
                n_workers=4,
                timeout=3,
                problem_file=str(problem_file),
            )["pass@1"]
        )

    summary: dict[str, Any] = {
        "tasks": len(prompts),
        "max_new_tokens": max_new_tokens,
        "greedy_baseline": greedy_summary,
        "speculative": {},
    }

    for k in draft_ks:
        total_tokens = 0
        total_forwards = 0
        total_steps = 0
        total_proposed = 0
        total_accepted = 0
        acc_by_pos = [0] * k
        prop_by_pos = [0] * k
        total_sec = 0.0
        exact_matches = 0
        spec_tokens_list: list[list[int]] = []

        for idx, prompt in enumerate(prompts):
            ids = torch.tensor(
                [tokenizer(prompt, add_special_tokens=False)["input_ids"]],
                device=device,
            )
            out = model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_assistant_tokens=k,
            )
            toks = out[0, ids.size(1) :].tolist()
            st = model.last_gen_stats
            spec_tokens_list.append(toks)
            if toks == baseline_tokens[idx]:
                exact_matches += 1
            total_tokens += len(toks)
            total_forwards += st["target_forwards"]
            total_steps += st["draft_steps"]
            total_proposed += st["draft_tokens_proposed"]
            total_accepted += st["draft_tokens_accepted"]
            total_sec += st["elapsed_seconds"]
            for j in range(k):
                acc_by_pos[j] += st["accepted_by_pos"][j]
                prop_by_pos[j] += st["proposed_by_pos"][j]

        tps = total_tokens / max(total_sec, 1e-9)
        k_entry: dict[str, Any] = {
            "draft_k": k,
            "total_tokens": total_tokens,
            "target_forwards": total_forwards,
            "draft_steps": total_steps,
            "draft_tokens_proposed": total_proposed,
            "draft_tokens_accepted": total_accepted,
            "overall_acceptance_rate": total_accepted / max(total_proposed, 1),
            "mean_accepted_per_step": total_accepted / max(total_steps, 1),
            "tokens_per_target_forward": total_tokens / max(total_forwards, 1),
            "forward_pass_reduction_factor": base_total_tokens / max(total_forwards, 1),
            "acceptance_rate_by_pos": [
                acc_by_pos[j] / max(prop_by_pos[j], 1) for j in range(k)
            ],
            "elapsed_seconds": total_sec,
            "tokens_per_sec": tps,
            "speedup_vs_greedy": tps / max(base_tps, 1e-9),
            "exact_token_match_rate": exact_matches / max(len(prompts), 1),
        }
        if task_ids is not None and problem_file is not None and output_dir is not None:
            k_path = output_dir / f"spec_k{k}_samples.jsonl"
            with k_path.open("w") as f:
                for tid, toks in zip(task_ids, spec_tokens_list):
                    raw = tokenizer.decode(toks, skip_special_tokens=True)
                    f.write(
                        json.dumps(
                            {"task_id": tid, "completion": truncate_at_stops(raw)}
                        )
                        + "\n"
                    )
            k_entry["pass@1"] = float(
                evaluate_functional_correctness(
                    str(k_path),
                    k=[1],
                    n_workers=4,
                    timeout=3,
                    problem_file=str(problem_file),
                )["pass@1"]
            )
        summary["speculative"][f"k={k}"] = k_entry

    model.config.use_cache = was_cache
    return summary


def evaluate_model(config: EvalConfig) -> dict[str, Any]:
    """Validate the harness, generate completions, then score with HumanEval."""
    import torch

    if config.output.exists():
        raise FileExistsError(f"{config.output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    model, tokenizer, adapter_hashes = load_evaluation_model(config)
    return run_benchmark_eval(model, tokenizer, config, adapter_hashes)


def main() -> None:
    config = parse_args()
    summary = evaluate_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
