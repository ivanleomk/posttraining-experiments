"""Answer-only GPT-J LoRA SFT on one GPU; dataset code is never executed.
Install: pip install torch==2.8.0 transformers==4.56.2 peft==0.17.1 pydantic==2.12.3
Run: python train_lora.py --train /tmp/code-splits/train-100.jsonl --output /tmp/lora-100
Start each dataset-size arm from the base model. Test data is not used in training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FilePath, ValidationError

MODEL_ID = "EleutherAI/gpt-j-6b"
MODEL_REVISION = "47e169305d2e8376be1d31e765533382721b2cc1"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train: FilePath
    output: Path
    epochs: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0, allow_inf_nan=False)
    accumulation: int = Field(default=4, ge=1)
    seed: int = 42


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    try:
        return TrainConfig.model_validate(vars(p.parse_args(argv)))
    except ValidationError as error:
        p.error(str(error))


def load_training_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(r.get("split", "train") != "train" for r in rows):
        raise ValueError("Expected nonempty training rows only")
    return rows


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


def run_training_loop(
    model: Any, batches: list[tuple[list[int], list[int]]], config: TrainConfig
) -> int:
    import torch
    from transformers import get_cosine_schedule_with_warmup

    params = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)
    total_steps = math.ceil(len(batches) / config.accumulation) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 1, total_steps)
    rng = random.Random(config.seed)
    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)
    with (config.output / "metrics.jsonl").open("w") as metrics:
        for epoch in range(config.epochs):
            order = list(range(len(batches)))
            rng.shuffle(order)
            for offset in range(0, len(order), config.accumulation):
                group = [
                    batches[i] for i in order[offset : offset + config.accumulation]
                ]
                counts = [sum(t != -100 for t in labels[1:]) for _, labels in group]
                loss_value = 0.0
                for (ids, labels), count in zip(group, counts):
                    loss = model(
                        input_ids=torch.tensor([ids], device="cuda"),
                        labels=torch.tensor([labels], device="cuda"),
                    ).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss")
                    weight = count / sum(counts)
                    (loss * weight).backward()
                    loss_value += float(loss.detach()) * weight
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite gradients")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                record = dict(
                    step=step,
                    total_steps=total_steps,
                    epoch=epoch + 1,
                    answer_token_nll=loss_value,
                )
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                print(json.dumps(record), flush=True)
    return step


def train_model(config: TrainConfig) -> dict[str, Any]:
    """Load data and a fresh base model, train LoRA, and save the adapter."""
    import torch
    from transformers import AutoTokenizer, set_seed

    if config.output.exists():
        raise FileExistsError(f"{config.output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    set_seed(config.seed)
    rows = load_training_rows(config.train)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    batches = tokenize_training_rows(rows, tokenizer)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = load_lora_model(dtype)
    config.output.mkdir(parents=True)
    metadata = dict(
        config.model_dump(mode="json"),
        model=MODEL_ID,
        revision=MODEL_REVISION,
        train_sha256=hashlib.sha256(config.train.read_bytes()).hexdigest(),
        examples=len(rows),
        rank=8,
        alpha=16,
        target_modules=["q_proj", "v_proj"],
        dtype=str(dtype),
    )
    (config.output / "config.json").write_text(json.dumps(metadata, indent=2))
    steps = run_training_loop(model, batches, config)
    model.save_pretrained(config.output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(config.output / "adapter")
    summary = dict(status="complete", steps=steps, config=metadata)
    (config.output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    config = parse_args()
    train_model(config)


if __name__ == "__main__":
    main()
