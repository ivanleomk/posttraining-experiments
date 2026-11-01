"""HumanEval on GPT-J, optionally loading a LoRA adapter. Run on an isolated GPU sandbox.
Install: pip install torch==2.8.0 transformers==4.56.2 peft==0.17.1
Install official evaluator:
  pip install https://github.com/openai/human-eval/archive/b9453e8f1865df0a0057a1383767caa400acfbcf.zip
Base: python evaluate.py --output /tmp/base-eval --execute-in-sandbox
LoRA: python evaluate.py --adapter /tmp/lora-100/adapter --output /tmp/lora-eval --execute-in-sandbox
Executes generated Python: use a disposable sandbox without credentials or host mounts.
Defaults match our earlier run: 20 samples/task, temperature .2, top_p .95,
top_k 0, max 512 new tokens, batch 10, seed 42; reports pass@1 and pass@10.
This is a reduced-sample protocol, not a full replication of the Codex paper.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

STOPS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--execute-in-sandbox', action='store_true')
    a = p.parse_args()
    if not a.execute_in_sandbox:
        p.error('Run inside an isolated sandbox and pass --execute-in-sandbox')
    if a.samples < 10 or a.batch_size < 1 or a.max_new_tokens < 1:
        p.error('At least 10 samples and positive batch size/token cap required')
    if a.output.exists():
        p.error('output already exists; choose a new run directory')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed
    from human_eval.data import read_problems
    from human_eval.evaluation import evaluate_functional_correctness
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    problems = read_problems()
    a.output.mkdir(parents=True)
    problem_file = a.output / 'problems.jsonl'
    problem_file.write_text(''.join(json.dumps(dict(v, task_id=k)) + '\n' for k, v in problems.items()))
    canonical = a.output / 'canonical.jsonl'
    canonical.write_text(''.join(json.dumps(dict(task_id=k, completion=v['canonical_solution'])) + '\n'
                                 for k, v in problems.items()))
    control = evaluate_functional_correctness(str(canonical), k=[1], n_workers=4, timeout=3,
                                             problem_file=str(problem_file))
    if control['pass@1'] != 1:
        raise RuntimeError(f'Canonical control failed: {control}')
    model_id = 'EleutherAI/gpt-j-6b'
    revision = '47e169305d2e8376be1d31e765533382721b2cc1'
    config = dict(vars(a), output=str(a.output), adapter=str(a.adapter) if a.adapter else None,
                  model=model_id, revision=revision, temperature=.2, top_p=.95, top_k=0,
                  stops=STOPS, torch=torch.__version__, transformers=transformers.__version__,
                  gpu=torch.cuda.get_device_name(0),
                  dataset_sha256=hashlib.sha256(json.dumps(problems, sort_keys=True).encode()).hexdigest())
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16,
                                                use_safetensors=False, trust_remote_code=False).to('cuda').eval()
    if a.adapter:
        from peft import PeftModel
        adapter_config = json.loads((a.adapter / 'adapter_config.json').read_text())
        if adapter_config['base_model_name_or_path'] != model_id:
            raise ValueError('Adapter base model differs from GPT-J')
        config['adapter_files_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                          for f in a.adapter.iterdir() if f.is_file()}
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    (a.output / 'config.json').write_text(json.dumps(config, indent=2))

    class Stop(StoppingCriteria):
        def __init__(self, start):
            self.start = start

        def __call__(self, input_ids, scores, **kwargs):
            texts = tok.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
            return torch.tensor([any(s in text for s in STOPS) for text in texts], device=input_ids.device)

    start = time.monotonic()
    cap_count = 0
    sample_path = a.output / 'samples.jsonl'
    with sample_path.open('w') as samples, (a.output / 'raw.jsonl').open('w') as raw:
        for index, (task_id, problem) in enumerate(problems.items()):
            inputs = tok(problem['prompt'], return_tensors='pt').to('cuda')
            prompt_len = inputs['input_ids'].shape[1]
            if prompt_len + a.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError(f'Context overflow: {task_id}')
            for offset in range(0, a.samples, a.batch_size):
                set_seed(a.seed + index * 1000 + offset)
                n = min(a.batch_size, a.samples - offset)
                with torch.inference_mode():
                    output = model.generate(**{k: v.repeat(n, 1) for k, v in inputs.items()},
                                            do_sample=True, temperature=.2, top_p=.95, top_k=0,
                                            max_new_tokens=a.max_new_tokens, pad_token_id=tok.eos_token_id,
                                            stopping_criteria=StoppingCriteriaList([Stop(prompt_len)]))
                for seq in output[:, prompt_len:]:
                    text = tok.decode(seq, skip_special_tokens=True)
                    stops = [text.find(s) for s in STOPS if s in text]
                    hit_cap = len(seq) == a.max_new_tokens and not stops and tok.eos_token_id not in seq.tolist()
                    cap_count += int(hit_cap)
                    raw.write(json.dumps(dict(task_id=task_id, completion=text, hit_output_cap=hit_cap)) + '\n')
                    samples.write(json.dumps(dict(task_id=task_id, completion=text[:min(stops)] if stops else text)) + '\n')
                raw.flush()
                samples.flush()
            print(json.dumps(dict(tasks_completed=index + 1, tasks_total=len(problems))), flush=True)
    scores = evaluate_functional_correctness(str(sample_path), k=[1, 10], n_workers=4, timeout=3,
                                            problem_file=str(problem_file))
    summary = dict(status='complete', config=config, scores=scores, canonical_control=control,
                   output_cap_hits=cap_count, elapsed_seconds=time.monotonic() - start)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
