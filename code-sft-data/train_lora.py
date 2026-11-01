"""Answer-only GPT-J LoRA SFT on one GPU; dataset code is never executed.
Install: pip install torch==2.8.0 transformers==4.56.2 peft==0.17.1
Run: python train_lora.py --train /tmp/code-splits/train-100.jsonl --output /tmp/lora-100
Start each dataset-size arm from the base model. Test data is not used in training.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.epochs < 1 or a.accumulation < 1 or a.learning_rate <= 0:
        p.error('epochs, accumulation and learning rate must be positive')
    if a.output.exists():
        p.error('output already exists; choose a new run directory')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    set_seed(a.seed)
    rows = [json.loads(line) for line in a.train.read_text().splitlines() if line.strip()]
    if not rows or any(r.get('split', 'train') != 'train' for r in rows):
        raise ValueError('Expected nonempty training rows only')
    model_id = 'EleutherAI/gpt-j-6b'
    revision = '47e169305d2e8376be1d31e765533382721b2cc1'
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    batches = []
    for row in rows:
        boundary = len(row['prompt'])
        e = tok(row['prompt'] + row['completion'], return_offsets_mapping=True, add_special_tokens=False)
        ids = e['input_ids'] + [tok.eos_token_id]
        labels = [t if start >= boundary and end > start else -100
                  for t, (start, end) in zip(e['input_ids'], e['offset_mapping'])] + [tok.eos_token_id]
        if len(ids) > 2048 or not any(t != -100 for t in labels[1:]):
            raise ValueError('Invalid sequence; refusing truncation')
        batches.append((ids, labels))
    a.output.mkdir(parents=True)
    config = dict(vars(a), train=str(a.train), output=str(a.output), model=model_id, revision=revision,
                  train_sha256=hashlib.sha256(a.train.read_bytes()).hexdigest(), examples=len(rows),
                  rank=8, alpha=16, target_modules=['q_proj', 'v_proj'], dtype=str(dtype))
    (a.output / 'config.json').write_text(json.dumps(config, indent=2))
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype,
                                                use_safetensors=False, trust_remote_code=False).to('cuda')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16,
                                           lora_dropout=0., target_modules=['q_proj', 'v_proj'], bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    params = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.learning_rate, weight_decay=.01)
    total_steps = math.ceil(len(batches) / a.accumulation) * a.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 1, total_steps)
    rng = random.Random(a.seed)
    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)
    with (a.output / 'metrics.jsonl').open('w') as metrics:
        for epoch in range(a.epochs):
            order = list(range(len(batches)))
            rng.shuffle(order)
            for offset in range(0, len(order), a.accumulation):
                group = [batches[i] for i in order[offset:offset + a.accumulation]]
                counts = [sum(t != -100 for t in labels[1:]) for _, labels in group]
                loss_value = 0.
                for (ids, labels), count in zip(group, counts):
                    loss = model(input_ids=torch.tensor([ids], device='cuda'),
                                 labels=torch.tensor([labels], device='cuda')).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    weight = count / sum(counts)
                    (loss * weight).backward()
                    loss_value += float(loss.detach()) * weight
                norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite gradients')
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                record = dict(step=step, total_steps=total_steps, epoch=epoch + 1, answer_token_nll=loss_value)
                metrics.write(json.dumps(record) + '\n')
                metrics.flush()
                print(json.dumps(record), flush=True)
    model.save_pretrained(a.output / 'adapter', safe_serialization=True)
    tok.save_pretrained(a.output / 'adapter')
    (a.output / 'summary.json').write_text(json.dumps(dict(status='complete', steps=step, config=config), indent=2))


if __name__ == '__main__':
    main()
