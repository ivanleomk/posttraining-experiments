"""Answer-only GPT-J SFT (full fine-tuning or LoRA) on one GPU; dataset code is never executed.
Install: pip install torch==2.8.0 transformers==4.56.2 peft==0.17.1 bitsandbytes pydantic==2.12.3
Run full SFT: python train_lora.py --mode full --train /tmp/code-splits/train-1000.jsonl --val /tmp/code-splits/val-100.jsonl --output /tmp/full-1000
Start each dataset-size arm from the base model. Test data is not used in training.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FilePath, ValidationError

MODEL_ID = "EleutherAI/gpt-j-6b"
MODEL_REVISION = "47e169305d2e8376be1d31e765533382721b2cc1"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train: FilePath | None = None
    val: FilePath | None = None
    output: Path
    mode: Literal["full", "lora"] = "full"
    objective: Literal["ntp", "nextlat"] = "ntp"
    lambda_mse: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    lambda_kl: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    proj_factor: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    epochs: int = Field(default=1, ge=0)
    learning_rate: float = Field(default=1e-5, gt=0, allow_inf_nan=False)
    accumulation: int = Field(default=4, ge=1)
    val_every: int = Field(default=25, ge=1)
    eval_benchmarks: list[str] = Field(default_factory=list)
    save_checkpoint: bool = False
    hf_repo: str | None = None
    seed: int = 42


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path)
    p.add_argument("--val", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("full", "lora"), default="full")
    p.add_argument("--objective", choices=("ntp", "nextlat"), default="ntp")
    p.add_argument("--lambda-mse", type=float, default=1.0)
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--proj-factor", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument(
        "--eval-benchmark",
        dest="eval_benchmarks",
        action="append",
        default=[],
        help="Benchmark path or 'humaneval' to evaluate after training (repeatable)",
    )
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--hf-repo", type=str)
    p.add_argument("--seed", type=int, default=42)
    try:
        return TrainConfig.model_validate(vars(p.parse_args(argv)))
    except ValidationError as error:
        p.error(str(error))


def load_rows(path: Path, expected_split: str = "train") -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(r.get("split", expected_split) != expected_split for r in rows):
        raise ValueError(f"Expected nonempty {expected_split} rows only")
    return rows


def load_training_rows(path: Path) -> list[dict[str, Any]]:
    return load_rows(path, expected_split="train")


def tokenize_training_rows(
    rows: list[dict[str, Any]], tokenizer: Any
) -> list[tuple[list[int], list[int]]]:
    batches = []
    for row in rows:
        boundary = len(row["prompt"])
        e = tokenizer(
            row["prompt"] + row["completion"],
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        ids = e["input_ids"] + [tokenizer.eos_token_id]
        labels = [
            t if start >= boundary and end > start else -100
            for t, (start, end) in zip(e["input_ids"], e["offset_mapping"])
        ] + [tokenizer.eos_token_id]
        if len(ids) > 2048 or not any(t != -100 for t in labels[1:]):
            raise ValueError("Invalid sequence; refusing truncation")
        batches.append((ids, labels))
    return batches


def load_lora_model(dtype: Any) -> Any:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        use_safetensors=False,
        trust_remote_code=False,
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    return model


def load_full_model(dtype: Any, train_enabled: bool = True) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        use_safetensors=False,
        trust_remote_code=False,
    ).to("cuda")
    if train_enabled:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def build_optimizer(params: list[Any], config: TrainConfig) -> Any:
    import torch

    if config.mode == "full":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                params, lr=config.learning_rate, weight_decay=0.01
            )
        except ImportError:
            from transformers.optimization import Adafactor

            return Adafactor(
                params,
                lr=config.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=0.01,
            )
    return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)


def compute_answer_nll(
    model: Any, batches: list[tuple[list[int], list[int]]]
) -> float:
    import torch

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for ids, labels in batches:
            count = sum(t != -100 for t in labels[1:])
            loss = model(
                input_ids=torch.tensor([ids], device="cuda"),
                labels=torch.tensor([labels], device="cuda"),
            ).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")
            total_loss += float(loss.detach()) * count
            total_tokens += count
    if was_training:
        model.train()
    return total_loss / max(total_tokens, 1)


def run_training_loop(
    model: Any,
    batches: list[tuple[list[int], list[int]]],
    config: TrainConfig,
    val_batches: list[tuple[list[int], list[int]]] | None = None,
    eos_token_id: int = 50256,
) -> tuple[int, list[dict[str, Any]]]:
    import torch
    from transformers import get_cosine_schedule_with_warmup

    history: list[dict[str, Any]] = []
    total_steps = (
        math.ceil(len(batches) / config.accumulation) * config.epochs
        if batches and config.epochs > 0
        else 0
    )
    with (config.output / "metrics.jsonl").open("w") as metrics:
        if val_batches:
            init_record = dict(
                step=0,
                total_steps=total_steps,
                epoch=0,
                examples_seen=0,
                val_answer_token_nll=compute_answer_nll(model, val_batches),
            )
            history.append(init_record)
            metrics.write(json.dumps(init_record) + "\n")
            metrics.flush()
            print(json.dumps(init_record), flush=True)

        if total_steps == 0:
            return 0, history

        dynamics_model = None
        params = [v for v in model.parameters() if v.requires_grad]
        if config.objective == "nextlat":
            from nextlat import compute_nextlat_losses, create_dynamics_model

            dynamics_model = create_dynamics_model(
                n_embd=model.config.n_embd,
                proj_factor=config.proj_factor,
            ).to(device="cuda", dtype=next(model.parameters()).dtype)
            dynamics_model.train()
            params.extend(v for v in dynamics_model.parameters() if v.requires_grad)

        optimizer = build_optimizer(params, config)
        warmup_steps = max(1, int(0.05 * total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        rng = random.Random(config.seed)
        model.train()
        step = 0
        examples_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(config.epochs):
            order = list(range(len(batches)))
            rng.shuffle(order)
            for offset in range(0, len(order), config.accumulation):
                group = [
                    batches[i] for i in order[offset : offset + config.accumulation]
                ]
                counts = [sum(t != -100 for t in labels[1:]) for _, labels in group]
                loss_value = 0.0
                mse_value = 0.0
                kl_value = 0.0
                total_loss_value = 0.0
                for (ids, labels), count in zip(group, counts):
                    weight = count / sum(counts)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        if dynamics_model is not None:
                            losses = compute_nextlat_losses(
                                model,
                                dynamics_model,
                                input_ids=torch.tensor([ids], device="cuda"),
                                labels=torch.tensor([labels], device="cuda"),
                                eos_token_id=eos_token_id,
                            )
                            ntp_loss = losses["ntp_loss"]
                            mse_loss = losses["mse_loss"]
                            kl_loss = losses["kl_loss"]
                            loss = (
                                ntp_loss
                                + config.lambda_mse * mse_loss
                                + config.lambda_kl * kl_loss
                            )
                            mse_value += float(mse_loss.detach()) * weight
                            kl_value += float(kl_loss.detach()) * weight
                        else:
                            ntp_loss = model(
                                input_ids=torch.tensor([ids], device="cuda"),
                                labels=torch.tensor([labels], device="cuda"),
                            ).loss
                            loss = ntp_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss")
                    (loss * weight).backward()
                    loss_value += float(ntp_loss.detach()) * weight
                    total_loss_value += float(loss.detach()) * weight
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite gradients")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                examples_seen += len(group)
                record: dict[str, Any] = dict(
                    step=step,
                    total_steps=total_steps,
                    epoch=epoch + 1,
                    examples_seen=examples_seen,
                    answer_token_nll=loss_value,
                    grad_norm=float(norm),
                    learning_rate=float(scheduler.get_last_lr()[0]),
                )
                if dynamics_model is not None:
                    record.update(
                        nextlat_mse_loss=mse_value,
                        nextlat_kl_loss=kl_value,
                        total_loss=total_loss_value,
                    )
                if val_batches and (
                    step % config.val_every == 0 or step == total_steps
                ):
                    record["val_answer_token_nll"] = compute_answer_nll(
                        model, val_batches
                    )
                history.append(record)
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                print(json.dumps(record), flush=True)
        del dynamics_model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return step, history


def run_post_training_evals(
    model: Any, tokenizer: Any, config: TrainConfig
) -> dict[str, Any]:
    import torch
    from evaluate import EvalConfig, run_benchmark_eval

    if not config.eval_benchmarks:
        return {}
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model = model.to(torch.bfloat16).eval()
    tokenizer.pad_token = tokenizer.eos_token
    gc.collect()
    torch.cuda.empty_cache()

    results: dict[str, Any] = {}
    for bench in config.eval_benchmarks:
        is_humaneval = bench.lower() == "humaneval"
        bench_path = None if is_humaneval else Path(bench)
        name = "humaneval" if is_humaneval else bench_path.stem.replace("-", "_")
        eval_cfg = EvalConfig(
            output=config.output / f"eval_{name}",
            benchmark=bench_path,
            samples=20,
            batch_size=10,
            max_new_tokens=512,
            seed=config.seed,
            execute_in_sandbox=True,
        )
        results[name] = run_benchmark_eval(model, tokenizer, eval_cfg)
    return results


def train_model(config: TrainConfig) -> dict[str, Any]:
    """Load data and a fresh base model, run SFT, and optionally evaluate."""
    import torch
    from transformers import AutoTokenizer, set_seed

    if config.output.exists():
        raise FileExistsError(f"{config.output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    train_rows = load_training_rows(config.train) if config.train else []
    val_rows = load_rows(config.val, expected_split="val") if config.val else []
    train_batches = (
        tokenize_training_rows(train_rows, tokenizer) if train_rows else []
    )
    val_batches = tokenize_training_rows(val_rows, tokenizer) if val_rows else None

    if config.mode == "lora":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        model = load_lora_model(dtype)
    else:
        dtype = torch.float32 if config.epochs > 0 else torch.bfloat16
        model = load_full_model(dtype, train_enabled=config.epochs > 0)

    config.output.mkdir(parents=True)
    metadata = dict(
        config.model_dump(mode="json"),
        model=MODEL_ID,
        revision=MODEL_REVISION,
        train_sha256=(
            hashlib.sha256(config.train.read_bytes()).hexdigest()
            if config.train
            else None
        ),
        val_sha256=(
            hashlib.sha256(config.val.read_bytes()).hexdigest()
            if config.val
            else None
        ),
        examples=len(train_rows),
        val_examples=len(val_rows),
        dtype=str(dtype),
    )
    if config.mode == "lora":
        metadata.update(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    (config.output / "config.json").write_text(json.dumps(metadata, indent=2))

    steps, history = run_training_loop(
        model, train_batches, config, val_batches, eos_token_id=tokenizer.eos_token_id
    )

    if config.mode == "lora":
        model.save_pretrained(config.output / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(config.output / "adapter")
    elif config.save_checkpoint:
        model.to(torch.bfloat16).save_pretrained(
            config.output / "checkpoint", safe_serialization=True
        )
        tokenizer.save_pretrained(config.output / "checkpoint")

    if config.hf_repo:
        model.to(torch.bfloat16).push_to_hub(config.hf_repo, safe_serialization=True)
        tokenizer.push_to_hub(config.hf_repo)

    evaluations = run_post_training_evals(model, tokenizer, config)
    summary = dict(
        status="complete",
        steps=steps,
        config=metadata,
        final_metrics=history[-1] if history else None,
        evaluations={k: v["scores"] for k, v in evaluations.items()},
    )
    (config.output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    config = parse_args()
    train_model(config)


if __name__ == "__main__":
    main()
