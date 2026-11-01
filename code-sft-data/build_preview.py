"""Static extraction only. Never executes dataset code on the host."""
import ast, gzip, hashlib, json, re, urllib.request
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
REV='356bb069eee815daa6e23e9a282eeefe1490ad44'
MODEL_REV='47e169305d2e8376be1d31e765533382721b2cc1'
def digest(x): return hashlib.sha256(x.encode()).hexdigest()
def write(name, rows): (ROOT/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
def extract(row):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',row['response'],re.S)
    if len(blocks)!=1: raise ValueError('not_one_code_block')
    try: tree=ast.parse(blocks[0])
    except SyntaxError: raise ValueError('syntax_error')
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef): raise ValueError('not_one_standalone_function')
    fn=tree.body[0]
    if fn.decorator_list: raise ValueError('decorated_function')
    if any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.ClassDef,ast.AsyncFunctionDef)) for n in ast.walk(fn)): raise ValueError('complex_or_external_dependencies')
    calls={n.func.id for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    if calls & {'open','exec','eval','compile','__import__','input','globals','locals'}: raise ValueError('io_or_dynamic_execution')
    # Deterministic representation: same instruction, no teacher explanation, no seed code.
    if ast.get_docstring(fn) is not None: fn.body=fn.body[1:]
    if not fn.body: raise ValueError('empty_function')
    body='\n'.join(ast.unparse(n) for n in fn.body)
    signature=ast.unparse(fn).splitlines()[0]
    prompt=signature+'\n    '+repr(row['instruction'])+'\n'
    completion=''.join('    '+line+'\n' if line else '\n' for line in body.splitlines())
    return prompt,completion,digest(ast.dump(ast.Module(body=fn.body,type_ignores=[]),include_attributes=False))

def main():
    tok=AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b',revision=MODEL_REV)
    url='https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz'
    benchmark_file=ROOT/'humaneval-reference.jsonl.gz'
    if not benchmark_file.exists(): benchmark_file.write_bytes(urllib.request.urlopen(url).read())
    benchmark=[json.loads(l) for l in gzip.decompress(benchmark_file.read_bytes()).decode().splitlines()]
    # Conservative exact task-text/body checks, not a claim of semantic decontamination.
    eval_texts={' '.join(p['prompt'].split()).lower() for p in benchmark}
    eval_bodies=set()
    for p in benchmark:
        tree=ast.parse(p['prompt']+p['canonical_solution']); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
        if ast.get_docstring(fn) is not None: fn.body=fn.body[1:]
        eval_bodies.add(digest(ast.dump(ast.Module(body=fn.body,type_ignores=[]),include_attributes=False)))
    rows=pq.read_table(ROOT/'source.parquet').slice(0,500).to_pylist()
    rejected=[]; accepted=[]; seen=set()
    for index,row in enumerate(rows):
        try:
            prompt,completion,body_hash=extract(row)
            if body_hash in eval_bodies or ' '.join(row['instruction'].split()).lower() in eval_texts: raise ValueError('exact_benchmark_overlap')
            if body_hash in seen: raise ValueError('duplicate_body')
            if len(tok(prompt+completion)['input_ids'])+1>2048: raise ValueError('context_too_long')
            seen.add(body_hash)
            split='dev' if int(digest(row['sha1'])[:8],16)%10==0 else 'train'
            accepted.append(dict(source_id=row['id'],source_row=index,source_revision=REV,source_seed_sha1=row['sha1'],source_url=f'https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k/tree/{REV}',instruction=row['instruction'],original_response=row['response'],prompt=prompt,completion=completion,token_count=len(tok(prompt+completion)['input_ids'])+1,body_sha256=body_hash,split=split,validation_status='static_candidate_not_execution_verified',loss_mask='prompt=-100; completion+EOS=supervised'))
        except ValueError as exc: rejected.append(dict(source_id=row['id'],source_row=index,reason=str(exc)))
    # First 500 source rows are a transparent inspection sample, not a representative corpus.
    selected=accepted[:100]
    write('candidates.jsonl',selected); write('rejected.jsonl',rejected)
    write('candidate-train.jsonl',[r for r in selected if r['split']=='train'])
    write('candidate-dev.jsonl',[r for r in selected if r['split']=='dev'])
    ids={165808,533341,81575}
    write('reviewed-preview.jsonl',[dict(r,validation_status='human_reviewed_tests_pending') for r in accepted if r['source_id'] in ids])
    report=dict(dataset='bigcode/self-oss-instruct-sc2-exec-filter-50k',revision=REV,license='odc-by',source_file_sha256=hashlib.sha256((ROOT/'source.parquet').read_bytes()).hexdigest(),source_rows_total=pq.read_metadata(ROOT/'source.parquet').num_rows,inspected_rows=500,static_accepted=len(accepted),selected_candidates=len(selected),candidate_splits=dict(Counter(r['split'] for r in selected)),rejection_reasons=dict(Counter(r['reason'] for r in rejected)),tokenizer_revision=MODEL_REV,benchmark_sha256=hashlib.sha256(benchmark_file.read_bytes()).hexdigest(),notes=['Static candidates are not approved for training.','Exact text/AST screening does not establish absence of paraphrased benchmark contamination.','Split is grouped by source seed SHA1; near-duplicate families need further review.','No code was executed on the host.','Prompt uses the exact source instruction as a Python docstring; body is AST-normalized.'])
    (ROOT/'manifest.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
