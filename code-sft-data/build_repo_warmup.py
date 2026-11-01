"""Build a decontaminated Python repository warmup corpus for NextLat dynamics warmup / CPT.

Extracts standalone, documented synchronous Python functions from local Python repositories
(including CPython's standard library and cloned open-source repositories), filters out
functions with unresolved names or unsafe dynamic execution calls, and removes any canonical
AST body or docstring overlaps with HumanEval and the frozen splits-v1 dataset.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import re
import sysconfig
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from generate_splits import (
    MODEL_REV,
    ROOT,
    canonical_body,
    digest,
    text_key,
    unresolved_names,
)


class RepoWarmupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    repos_dir: Path = Path("/tmp/python-repos")
    splits_dir: Path
    output: Path
    size: int = Field(default=1000, ge=1)
    seed: int = 42


def parse_args(argv: list[str] | None = None) -> RepoWarmupConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos-dir", type=Path, default=Path("/tmp/python-repos"))
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    try:
        return RepoWarmupConfig.model_validate(vars(p.parse_args(argv)))
    except ValidationError as error:
        p.error(str(error))


def load_blocked_hashes(splits_dir: Path) -> tuple[set[str], set[str], set[str]]:
    blocked_bodies: set[str] = set()
    blocked_tasks: set[str] = set()
    blocked_names: set[str] = set()

    he_path = ROOT / "humaneval-reference.jsonl.gz"
    with gzip.open(he_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            blocked_names.add(row["entry_point"].lower())
            try:
                blocked_bodies.add(
                    canonical_body(row["prompt"] + row["canonical_solution"])
                )
            except Exception:
                pass

    for split_file in ("train-10000.jsonl", "val-100.jsonl", "test-100.jsonl"):
        p = splits_dir / split_file
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "canonical_body_sha256" in row:
                blocked_bodies.add(row["canonical_body_sha256"])
            if "instruction_sha256" in row:
                blocked_tasks.add(row["instruction_sha256"])
            try:
                blocked_bodies.add(
                    canonical_body(row["prompt"] + row["completion"])
                )
            except Exception:
                pass
    return blocked_bodies, blocked_tasks, blocked_names


def extract_repo_function(node: ast.FunctionDef) -> tuple[str, str, str, str] | None:
    if node.decorator_list:
        return None
    doc = ast.get_docstring(node)
    if not doc:
        return None
    clean_doc = doc.strip().splitlines()[0].strip()
    if len(clean_doc) < 15 or len(clean_doc) > 400:
        return None
    if any(
        isinstance(
            n,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.ClassDef,
                ast.AsyncFunctionDef,
            ),
        )
        for n in ast.walk(node)
    ):
        return None
    calls = {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    if calls & {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "globals",
        "locals",
    }:
        return None

    fn_copy = ast.parse(ast.unparse(node)).body[0]
    assert isinstance(fn_copy, ast.FunctionDef)
    if ast.get_docstring(fn_copy) is not None:
        fn_copy.body = fn_copy.body[1:]
    if not fn_copy.body:
        return None

    full_unparsed = ast.unparse(fn_copy)
    if len(full_unparsed) > 2500 or len(full_unparsed.splitlines()) < 3:
        return None
    if unresolved_names(full_unparsed):
        return None

    body_sha = canonical_body(full_unparsed)
    body = "\n".join(ast.unparse(n) for n in fn_copy.body)
    signature = full_unparsed.splitlines()[0]
    prompt = signature + "\n    " + repr(clean_doc) + "\n"
    completion = "".join(
        "    " + line + "\n" if line else "\n" for line in body.splitlines()
    )
    return prompt, completion, clean_doc, body_sha


def build_repo_warmup(config: RepoWarmupConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6b", revision=MODEL_REV
    )
    blocked_bodies, blocked_tasks, blocked_names = load_blocked_hashes(
        config.splits_dir
    )

    stdlib = Path(sysconfig.get_path("stdlib"))
    roots: list[tuple[str, Path]] = [("cpython-stdlib", stdlib)]
    if config.repos_dir.exists():
        for p in sorted(config.repos_dir.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                roots.append((p.name, p))

    candidates: list[dict[str, Any]] = []
    seen_bodies = set(blocked_bodies)
    seen_tasks = set(blocked_tasks)
    repo_counts: Counter[str] = Counter()

    for repo_name, root in roots:
        for py_file in sorted(root.rglob("*.py")):
            if any(
                part in ("test", "tests", "testing", "site-packages", "venv", ".git")
                for part in py_file.parts
            ):
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except Exception:
                continue
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.name.startswith("_") or node.name.lower() in blocked_names:
                    continue
                try:
                    extracted = extract_repo_function(node)
                except Exception:
                    continue
                if extracted is None:
                    continue
                prompt, completion, clean_doc, body_sha = extracted
                task_sha = digest(text_key(clean_doc))
                if body_sha in seen_bodies or task_sha in seen_tasks:
                    continue
                tokens = tokenizer(
                    prompt + completion, add_special_tokens=False
                )["input_ids"]
                if len(tokens) + 1 > 1024:
                    continue
                seen_bodies.add(body_sha)
                seen_tasks.add(task_sha)
                rel_path = str(py_file.relative_to(root))
                candidates.append(
                    {
                        "id": f"repo/{repo_name}/{rel_path}:{node.name}",
                        "split": "train",
                        "repo": repo_name,
                        "path": rel_path,
                        "entry_point": node.name,
                        "prompt": prompt,
                        "completion": completion,
                        "canonical_body_sha256": body_sha,
                        "instruction_sha256": task_sha,
                    }
                )
                repo_counts[repo_name] += 1

    candidates.sort(key=lambda r: digest(f"{config.seed}:{r['id']}"))
    if len(candidates) < config.size:
        raise ValueError(
            f"Only found {len(candidates)} eligible repo functions, needed {config.size}"
        )
    selected = candidates[: config.size]
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        "".join(json.dumps(r) + "\n" for r in selected), encoding="utf-8"
    )
    out_sha = hashlib.sha256(config.output.read_bytes()).hexdigest()
    selected_counts = Counter(r["repo"] for r in selected)
    summary = {
        "output": str(config.output),
        "size": len(selected),
        "total_candidates": len(candidates),
        "sha256": out_sha,
        "repos": dict(selected_counts),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    config = parse_args()
    build_repo_warmup(config)


if __name__ == "__main__":
    main()
