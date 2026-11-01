"""Build pinned, nested code-SFT pools. Parses code but never executes it."""
import argparse, ast, builtins, copy, gzip, hashlib, json, os, random, re, shutil, symtable, tempfile
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from build_preview import extract, REV, MODEL_REV
ROOT=Path(__file__).resolve().parent
SOURCE_SHA256='7bd6092e9940aedebe31f7049c460ba1549a920de5cb4b9bb36b0093cd51fc93'
BLOCKED_IDS={533341}

def digest(text): return hashlib.sha256(text.encode()).hexdigest()
def text_key(text): return re.sub(r'\W+',' ',text.lower()).strip()

def function_info(source):
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    if ast.get_docstring(fn) is not None: fn.body=fn.body[1:]
    return tree,fn

def canonical_body(source):
    _,fn=function_info(source)
    names={fn.name:'SELF'}
    for arg in [*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs]: names.setdefault(arg.arg,f'v{len(names)}')
    for arg in [fn.args.vararg,fn.args.kwarg]:
        if arg: names.setdefault(arg.arg,f'v{len(names)}')
    for n in ast.walk(fn):
        if isinstance(n,ast.Name) and isinstance(n.ctx,(ast.Store,ast.Del)): names.setdefault(n.id,f'v{len(names)}')
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            node.id=names.get(node.id,node.id); return node
        def visit_arg(self,node):
            node.arg=names.get(node.arg,node.arg); return node
    body=Rename().visit(ast.Module(body=copy.deepcopy(fn.body),type_ignores=[]))
    return digest(ast.dump(body,include_attributes=False))

def unresolved_names(source):
    table=symtable.symtable(source,'<candidate>','exec')
    module_defined={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    allowed=set(dir(builtins))|module_defined
    unknown=set()
    def walk(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in allowed: unknown.add(symbol.get_name())
        for child in scope.get_children(): walk(child)
    walk(table)
    return sorted(unknown)

class Groups:
    def __init__(self): self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        while x!=self.parent[x]:
            self.parent[x]=self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.parent[max(a,b)]=min(a,b)

def choose(rows,train_size,test_size,seed):
    """Union related rows before deduplication, then reserve whole holdout families."""
    groups=Groups()
    for row in rows:
        keys=['seed:'+row['source_seed_sha1'],'body:'+row['canonical_body_sha256'],'task:'+row['instruction_sha256']]
        for key in keys[1:]: groups.union(keys[0],key)
    unique=[]; seen=set()
    ordered=sorted(rows,key=lambda r:(r['source_row'],str(r['source_id'])))
    for row in ordered:
        key=row['canonical_body_sha256']
        if key in seen: continue
        seen.add(key)
        row=dict(row,family_id=digest(groups.find('seed:'+row['source_seed_sha1'])))
        unique.append(row)
    unique.sort(key=lambda r:digest(f"{seed}:{r['source_id']}:{r['source_row']}"))
    holdout=unique[:test_size]
    held_families={r['family_id'] for r in holdout}
    train=[r for r in unique[test_size:] if r['family_id'] not in held_families][:train_size]
    if len(holdout)!=test_size or len(train)!=train_size: raise ValueError(f'Insufficient eligible examples: train={len(train)}, holdout={len(holdout)}')
    assert not ({r['family_id'] for r in train}&held_families)
    return train,holdout,dict(unique_candidates=len(unique),duplicate_bodies_removed=len(rows)-len(unique),holdout_families=len(held_families),unused_holdout_family_rows=sum(r['family_id'] in held_families for r in unique[test_size:]))

def write_jsonl(path,rows): path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def stats(rows):
    return dict(examples=len(rows),total_tokens=sum(r['token_count'] for r in rows),answer_tokens=sum(r['answer_token_count'] for r in rows),max_tokens=max(r['token_count'] for r in rows))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'source.parquet')
    parser.add_argument('--output',type=Path,default=ROOT/'splits-v1')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-size',type=int,default=10000)
    parser.add_argument('--test-size',type=int,default=100)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(f'{args.output} exists; choose a new version directory')
    if args.train_size<1000 or args.test_size<1: raise ValueError('train-size >=1000 and test-size >=1 required')
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash!=SOURCE_SHA256: raise ValueError('Source hash differs from pinned BigCode parquet')
    benchmark_file=ROOT/'humaneval-reference.jsonl.gz'
    benchmark_hash=hashlib.sha256(benchmark_file.read_bytes()).hexdigest()
    if benchmark_hash!='b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef': raise ValueError('Benchmark reference hash mismatch')
    benchmark=[json.loads(l) for l in gzip.decompress(benchmark_file.read_bytes()).decode().splitlines()]
    eval_bodies={canonical_body(p['prompt']+p['canonical_solution']) for p in benchmark}
    eval_texts={text_key(ast.get_docstring(next(n for n in ast.parse(p['prompt']+p['canonical_solution']).body if isinstance(n,ast.FunctionDef))) or '') for p in benchmark}
    rows=pq.read_table(args.source).to_pylist()
    candidates=[]; rejected=[]
    tok=AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b',revision=MODEL_REV)
    for i,row in enumerate(rows):
        try:
            if row['id'] in BLOCKED_IDS: raise ValueError('known_contract_ambiguity')
            prompt,completion,body_hash=extract(row)
            unknown=unresolved_names(prompt+completion)
            if unknown: raise ValueError('unresolved_global_names')
            canonical=canonical_body(prompt+completion)
            if canonical in eval_bodies or text_key(row['instruction']) in eval_texts: raise ValueError('benchmark_overlap_normalized')
            candidates.append(dict(source_id=row['id'],source_row=i,source_revision=REV,source_seed_sha1=row['sha1'],instruction=row['instruction'],prompt=prompt,completion=completion,body_sha256=body_hash,canonical_body_sha256=canonical,instruction_sha256=digest(text_key(row['instruction'])),validation_status='upstream_execution_filtered_static_curated_not_individually_reverified'))
        except (ValueError,SyntaxError,StopIteration,RecursionError) as exc:
            reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
            rejected.append(dict(source_id=row['id'],source_row=i,reason=reason))
        if (i+1)%10000==0: print(f'Parsed {i+1}/{len(rows)}; static candidates {len(candidates)}',flush=True)
    accepted=[]
    for offset in range(0,len(candidates),256):
        batch=candidates[offset:offset+256]
        encoded=tok([r['prompt']+r['completion'] for r in batch],return_offsets_mapping=True,add_special_tokens=False,verbose=False)
        for row,ids,offsets in zip(batch,encoded['input_ids'],encoded['offset_mapping']):
            if len(ids)+1>2048:
                rejected.append(dict(source_id=row['source_id'],source_row=row['source_row'],reason='context_too_long')); continue
            boundary=len(row['prompt'])
            row['token_count']=len(ids)+1
            row['answer_token_count']=sum(a>=boundary and b>a for a,b in offsets)+1
            accepted.append(row)
    train,test,selection=choose(accepted,args.train_size,args.test_size,args.seed)
    for row in train: row['split']='train'
    for row in test: row['split']='test'
    pools={'train-100':train[:100],'train-1000':train[:1000],f'train-{args.train_size}':train,'test-100' if args.test_size==100 else f'test-{args.test_size}':test}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    staging=Path(tempfile.mkdtemp(prefix='.splits-',dir=args.output.parent))
    try:
        for name,pool in pools.items(): write_jsonl(staging/(name+'.jsonl'),pool)
        write_jsonl(staging/'rejected.jsonl',rejected)
        manifest=dict(dataset='bigcode/self-oss-instruct-sc2-exec-filter-50k',source_revision=REV,source_sha256=source_hash,license='odc-by',source_rows=len(rows),seed=args.seed,tokenizer='EleutherAI/gpt-j-6b',tokenizer_revision=MODEL_REV,context_limit=2048,benchmark_reference_sha256=benchmark_hash,static_candidates=len(candidates),length_filtered_candidates=len(accepted),selection=selection,rejection_reasons=dict(Counter(r['reason'] for r in rejected)),splits={name:stats(pool) for name,pool in pools.items()},file_sha256={name+'.jsonl':hashlib.sha256((staging/(name+'.jsonl')).read_bytes()).hexdigest() for name in pools},nested_train_subsets=True,holdout_role='Fixed in-domain test set; not HumanEval; never included in training.',validation_notes=['Upstream execution-filtered solutions; static processing does not certify reference correctness.','No dataset code executed locally.','Seed/task/canonical-body families are kept out of both train and test simultaneously.','Canonicalized exact overlap screening is not semantic decontamination.','Known ambiguous rectangle example 533341 is excluded.','Previous 93-example pilot is a separate experiment; it is not the 100-example arm of this new ladder.'])
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        os.rename(staging,args.output)
    except BaseException:
        shutil.rmtree(staging);raise
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
