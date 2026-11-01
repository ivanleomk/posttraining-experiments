"""Generate and verify executable check(candidate) tests using Gemini 3.8 Flash.
Run:
  python generate_tests.py --candidates /tmp/splits/val-200.jsonl --output /tmp/val-tests
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FilePath, ValidationError

from generate_splits import unresolved_names

PROMPT_VERSION = "v1.0"
SYSTEM_PROMPT = """You are writing an executable Python evaluation test suite for a coding benchmark.
1. Output one fenced Python block (```python ... ```) defining `def check(candidate):`.
2. Call the function under test exclusively through `candidate(...)`.
3. Use ONLY Python built-ins (no `import`, `open`, `input`, `print`, `exec`, `eval`, or `__import__`).
4. Include ordinary `assert` statements covering explicit examples in the description, typical inputs, and specified boundary cases.
5. Do NOT invent behavior for unspecified inputs, types, or error conditions.
6. Use `abs(actual - expected) < 1e-6` for non-exact floats, and `try: ... except Exception:` when testing required exceptions."""

FORBIDDEN_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "print",
}

CMP_SWAPS: dict[type[ast.cmpop], tuple[str, type[ast.cmpop]]] = {
    ast.Eq: ("eq_to_noteq", ast.NotEq),
    ast.NotEq: ("noteq_to_eq", ast.Eq),
    ast.Lt: ("lt_to_gt", ast.Gt),
    ast.LtE: ("lte_to_gt", ast.Gt),
    ast.Gt: ("gt_to_lt", ast.Lt),
    ast.GtE: ("gte_to_lt", ast.Lt),
    ast.Is: ("is_to_isnot", ast.IsNot),
    ast.IsNot: ("isnot_to_is", ast.Is),
    ast.In: ("in_to_notin", ast.NotIn),
    ast.NotIn: ("notin_to_in", ast.In),
}

BIN_SWAPS: dict[type[ast.operator], tuple[str, type[ast.operator]]] = {
    ast.Add: ("add_to_sub", ast.Sub),
    ast.Sub: ("sub_to_add", ast.Add),
    ast.Mult: ("mult_to_floordiv", ast.FloorDiv),
    ast.Div: ("div_to_mult", ast.Mult),
    ast.FloorDiv: ("floordiv_to_mult", ast.Mult),
    ast.Mod: ("mod_to_add", ast.Add),
}


class TestGenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidates: FilePath
    output: Path
    model: str = "models/gemini-3.8-flash"
    max_attempts: int = Field(default=1, ge=1)
    workers: int = Field(default=16, ge=1)
    api_key_file: FilePath = Path.home() / ".config/forge-experiments/gemini-api-key"


def parse_args(argv: list[str] | None = None) -> TestGenConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="models/gemini-3.8-flash")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path.home() / ".config/forge-experiments/gemini-api-key",
    )
    try:
        return TestGenConfig.model_validate(vars(parser.parse_args(argv)))
    except ValidationError as error:
        parser.error(str(error))


def build_test_prompt(row: dict[str, Any]) -> str:
    signature = row["prompt"].splitlines()[0]
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Function signature:\n```python\n{signature}\n```\n\n"
        f"Task instruction:\n{row['instruction']}\n"
    )


def extract_check_function(response_text: str) -> str:
    """Parse and validate a single standalone `def check(candidate):` block."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response_text, re.S)
    if len(blocks) != 1:
        raise ValueError("not_one_code_block")
    try:
        tree = ast.parse(blocks[0].strip())
    except SyntaxError as exc:
        raise ValueError("syntax_error") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("not_one_standalone_function")
    fn = tree.body[0]
    if (
        fn.name != "check"
        or fn.decorator_list
        or fn.args.posonlyargs
        or fn.args.kwonlyargs
        or fn.args.vararg
        or fn.args.kwarg
        or fn.args.defaults
        or [a.arg for a in fn.args.args] != ["candidate"]
    ):
        raise ValueError("invalid_check_signature")
    if any(
        isinstance(
            n,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.AsyncFunctionDef,
            ),
        )
        for n in ast.walk(fn)
    ):
        raise ValueError("complex_or_external_dependencies")
    calls = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    if calls & FORBIDDEN_CALLS:
        raise ValueError("io_or_dynamic_execution")
    if "candidate" not in calls:
        raise ValueError("candidate_not_called")
    if sum(isinstance(n, ast.Assert) for n in ast.walk(fn)) < 2:
        raise ValueError("insufficient_assertions")
    source = ast.unparse(tree) + "\n"
    if unresolved_names(source):
        raise ValueError("unresolved_global_names")
    return source


def build_mutants(prompt: str, completion: str) -> dict[str, dict[str, str]]:
    """Construct generic constant/identity mutants and AST-derived task mutants."""
    tree = ast.parse(prompt + completion)
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    if ast.get_docstring(fn) is not None:
        fn.body = fn.body[1:]
    sig = prompt.splitlines()[0]

    generic = {
        "const_none": f"{sig}\n    return None\n",
        "const_zero": f"{sig}\n    return 0\n",
        "const_one": f"{sig}\n    return 1\n",
        "const_empty_str": f"{sig}\n    return ''\n",
        "const_empty_list": f"{sig}\n    return []\n",
        "const_empty_dict": f"{sig}\n    return {{}}\n",
        "const_true": f"{sig}\n    return True\n",
        "const_false": f"{sig}\n    return False\n",
    }
    args = [*fn.args.posonlyargs, *fn.args.args]
    if args:
        generic["identity_first_arg"] = f"{sig}\n    return {args[0].arg}\n"

    ast_mutants: dict[str, str] = {}
    nodes = list(ast.walk(fn))
    for idx, node in enumerate(nodes):
        if len(ast_mutants) >= 8:
            break
        if isinstance(node, ast.Compare) and node.ops and type(node.ops[0]) in CMP_SWAPS:
            label, new_cmp = CMP_SWAPS[type(node.ops[0])]
            cloned = copy.deepcopy(fn)
            list(ast.walk(cloned))[idx].ops[0] = new_cmp()  # type: ignore[attr-defined]
            ast_mutants[f"ast_{label}_{idx}"] = ast.unparse(cloned) + "\n"
        elif isinstance(node, ast.BinOp) and type(node.op) in BIN_SWAPS:
            label, new_op = BIN_SWAPS[type(node.op)]
            cloned = copy.deepcopy(fn)
            list(ast.walk(cloned))[idx].op = new_op()  # type: ignore[attr-defined]
            ast_mutants[f"ast_{label}_{idx}"] = ast.unparse(cloned) + "\n"
        elif isinstance(node, ast.If):
            cloned = copy.deepcopy(fn)
            target = list(ast.walk(cloned))[idx]
            assert isinstance(target, ast.If)
            target.test = ast.UnaryOp(op=ast.Not(), operand=target.test)
            ast_mutants[f"ast_negate_if_{idx}"] = ast.unparse(cloned) + "\n"
        elif isinstance(node, ast.Return) and node.value is not None:
            for label, expr in (
                ("return_neg", ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(node.value))),
                (
                    "return_tail",
                    ast.Subscript(
                        value=copy.deepcopy(node.value),
                        slice=ast.Slice(lower=ast.Constant(value=1)),
                        ctx=ast.Load(),
                    ),
                ),
            ):
                cloned = copy.deepcopy(fn)
                list(ast.walk(cloned))[idx].value = expr  # type: ignore[attr-defined]
                ast_mutants[f"ast_{label}_{idx}"] = ast.unparse(cloned) + "\n"
    return {"generic": generic, "ast": ast_mutants}


def request_gemini(api_key: str, model: str, prompt: str, temperature: float) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate_row_test(row: dict[str, Any], config: TestGenConfig, api_key: str) -> dict[str, Any]:
    prompt = build_test_prompt(row)
    attempts: list[dict[str, Any]] = []
    parsed_test: str | None = None
    for attempt in range(1, config.max_attempts + 1):
        temperature = 0.0 if attempt == 1 else 0.2
        t0 = time.monotonic()
        raw_text = ""
        try:
            resp = request_gemini(api_key, config.model, prompt, temperature)
            candidates = resp.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            raw_text = "".join(p.get("text", "") for p in parts)
            parsed_test = extract_check_function(raw_text)
            attempts.append(
                dict(
                    attempt=attempt,
                    temperature=temperature,
                    elapsed_seconds=round(time.monotonic() - t0, 3),
                    usage=resp.get("usageMetadata"),
                    raw_response_text=raw_text,
                    outcome="valid_contract",
                )
            )
            break
        except Exception as exc:
            attempts.append(
                dict(
                    attempt=attempt,
                    temperature=temperature,
                    elapsed_seconds=round(time.monotonic() - t0, 3),
                    raw_response_text=raw_text,
                    outcome=f"error:{exc}",
                )
            )
            if isinstance(exc, urllib.error.HTTPError):
                time.sleep(2.0 * attempt)
    return dict(
        source_id=row["source_id"],
        task_id=f"bigcode/{row['source_id']}",
        model_id=config.model,
        prompt_version=PROMPT_VERSION,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        attempt_count=len(attempts),
        status="valid_contract" if parsed_test else "failed",
        parsed_test=parsed_test,
        attempts=attempts,
    )


def generate_all_tests(rows: list[dict[str, Any]], config: TestGenConfig) -> list[dict[str, Any]]:
    api_key = config.api_key_file.read_text().strip()
    config.output.mkdir(parents=True, exist_ok=True)
    gen_path = config.output / "generations.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if gen_path.exists():
        for line in gen_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "valid_contract":
                    completed[rec["source_id"]] = rec
    todo = [r for r in rows if r["source_id"] not in completed]
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(generate_row_test, r, config, api_key): r["source_id"] for r in todo}
        for idx, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            completed[rec["source_id"]] = rec
            ordered = [completed[r["source_id"]] for r in rows if r["source_id"] in completed]
            gen_path.write_text("".join(json.dumps(x) + "\n" for x in ordered))
            print(
                json.dumps(
                    dict(
                        completed=idx,
                        todo=len(todo),
                        source_id=rec["source_id"],
                        status=rec["status"],
                        attempts=rec["attempt_count"],
                    )
                ),
                flush=True,
            )
    return [completed[r["source_id"]] for r in rows]


def run_snippet(code: str, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode == 0, r.stderr[-400:] if r.stderr else ""
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"


def verify_candidates(
    rows: list[dict[str, Any]], generations: list[dict[str, Any]], output: Path
) -> list[dict[str, Any]]:
    by_id = {r["source_id"]: r for r in rows}
    out = []
    for gen in generations:
        if gen["status"] != "valid_contract" or not gen["parsed_test"]:
            continue
        row = by_id[gen["source_id"]]
        fn = ast.parse(row["prompt"] + row["completion"]).body[0]
        assert isinstance(fn, ast.FunctionDef)
        ep, test = fn.name, gen["parsed_test"]
        ref_code = row["prompt"] + row["completion"]
        mutants = build_mutants(row["prompt"], row["completion"])
        passed, stderr = run_snippet(f"{ref_code}\n{test}\ncheck({ep})\n", 3.0)
        gen_killed, ast_killed = [], []
        if passed:
            gen_killed = [
                k
                for k, m in mutants["generic"].items()
                if not run_snippet(f"{m}\n{test}\ncheck({ep})\n", 1.5)[0]
            ]
            ast_killed = [
                k
                for k, m in mutants["ast"].items()
                if not run_snippet(f"{m}\n{test}\ncheck({ep})\n", 1.5)[0]
            ]
        out.append(
            dict(
                source_id=row["source_id"],
                reference_passed=passed,
                reference_stderr=stderr,
                generic_total=len(mutants["generic"]),
                generic_killed=gen_killed,
                ast_killed=ast_killed,
            )
        )
    (output / "verification_results.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def generate_tests(config: TestGenConfig) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in config.candidates.read_text().splitlines()
        if line.strip()
    ]
    generations = generate_all_tests(rows, config)
    results = {
        r["source_id"]: r
        for r in verify_candidates(rows, generations, config.output)
    }
    gen_by_id = {g["source_id"]: g for g in generations}
    verified = []
    for row in rows:
        sid = row["source_id"]
        res = results.get(sid)
        if (
            res
            and res["reference_passed"]
            and len(res["generic_killed"]) == res["generic_total"]
            and len(res["ast_killed"]) >= 1
        ):
            fn = ast.parse(row["prompt"] + row["completion"]).body[0]
            assert isinstance(fn, ast.FunctionDef)
            verified.append(
                dict(
                    row,
                    task_id=f"bigcode/{sid}",
                    entry_point=fn.name,
                    canonical_solution=row["completion"],
                    test=gen_by_id[sid]["parsed_test"],
                )
            )
    (config.output / "verified.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in verified)
    )
    summary = dict(
        candidates=len(rows),
        valid_contracts=len(results),
        verified_tasks=len(verified),
    )
    (config.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    config = parse_args()
    print(json.dumps(generate_tests(config), indent=2))


if __name__ == "__main__":
    main()
