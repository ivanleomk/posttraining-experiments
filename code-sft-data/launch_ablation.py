"""Launch parallel GPU sandboxes for GPT-J full SFT dataset-size ablation with W&B tracking.
Run: /tmp/forge-gpu-sdk-venv/bin/python launch_ablation.py --splits-dir /Users/ivan/Documents/coding/experiments/code-sft-data/splits-v1 --output-dir /Users/ivan/Documents/coding/experiments/code-sft-data/runs/full-sft-ablation
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, ValidationError

CONTAINER_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
HUMAN_EVAL_COMMIT = "b9453e8f1865df0a0057a1383767caa400acfbcf"
HUMAN_EVAL_ZIP = (
    f"https://github.com/openai/human-eval/archive/{HUMAN_EVAL_COMMIT}.zip"
)


class ArmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    train_size: int
    train_file: FilePath | None = None
    warmup_file: FilePath | None = None
    warmup_epochs: int = 0
    warmup_lr: float = 1e-4
    cpt_epochs: int = 0
    cpt_lr: float = 5e-6
    zero_init_dynamics: bool = False
    lambda_mse: float | None = None
    lambda_kl: float | None = None
    epochs: int
    val_every: int
    learning_rate: float = 1e-5
    accumulation: int = 4


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    splits_dir: DirectoryPath
    output_dir: Path
    wandb_project: str = "gptj-code-sft"
    wandb_entity: str | None = None
    mode: str = "full"
    objective: str = "ntp"
    lambda_mse: float = Field(default=1.0, ge=0)
    lambda_kl: float = Field(default=1.0, ge=0)
    proj_factor: float = Field(default=0.5, gt=0)
    learning_rate: float = Field(default=1e-5, gt=0)
    poll_interval_seconds: int = Field(default=20, ge=5)
    max_lifetime_seconds: int = Field(default=10800, ge=600)
    seed: int = 42
    arms: list[str] = Field(
        default_factory=lambda: [
            "base",
            "full-sft-100",
            "full-sft-1000",
            "full-sft-10000",
        ]
    )


def parse_args(argv: list[str] | None = None) -> AblationConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--wandb-project", default="gptj-code-sft")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--mode", choices=("full", "lora"), default="full")
    p.add_argument("--objective", choices=("ntp", "nextlat"), default="ntp")
    p.add_argument("--lambda-mse", type=float, default=1.0)
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--proj-factor", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--poll-interval-seconds", type=int, default=20)
    p.add_argument("--max-lifetime-seconds", type=int, default=10800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--arms",
        nargs="+",
        default=["base", "full-sft-100", "full-sft-1000", "full-sft-10000"],
    )
    try:
        return AblationConfig.model_validate(vars(p.parse_args(argv)))
    except ValidationError as error:
        p.error(str(error))


def build_arms(config: AblationConfig) -> list[ArmSpec]:
    warmup_default = config.splits_dir / "repo-warmup-1000.jsonl"
    warmup_file = warmup_default if warmup_default.exists() else None
    specs = {
        "base": ArmSpec(
            name="base",
            train_size=0,
            train_file=None,
            epochs=0,
            val_every=1,
            learning_rate=config.learning_rate,
        ),
        "full-sft-100": ArmSpec(
            name="full-sft-100",
            train_size=100,
            train_file=config.splits_dir / "train-100.jsonl",
            epochs=1,
            val_every=5,
            learning_rate=config.learning_rate,
        ),
        "full-sft-1000": ArmSpec(
            name="full-sft-1000",
            train_size=1000,
            train_file=config.splits_dir / "train-1000.jsonl",
            epochs=1,
            val_every=25,
            learning_rate=config.learning_rate,
        ),
        "full-sft-10000": ArmSpec(
            name="full-sft-10000",
            train_size=10000,
            train_file=config.splits_dir / "train-10000.jsonl",
            epochs=1,
            val_every=100,
            learning_rate=config.learning_rate,
        ),
        "nextlat-sft-100": ArmSpec(
            name="nextlat-sft-100",
            train_size=100,
            train_file=config.splits_dir / "train-100.jsonl",
            epochs=1,
            val_every=5,
            learning_rate=config.learning_rate,
        ),
        "nextlat-sft-1000": ArmSpec(
            name="nextlat-sft-1000",
            train_size=1000,
            train_file=config.splits_dir / "train-1000.jsonl",
            epochs=1,
            val_every=25,
            learning_rate=config.learning_rate,
        ),
        "nextlat-sft-10000": ArmSpec(
            name="nextlat-sft-10000",
            train_size=10000,
            train_file=config.splits_dir / "train-10000.jsonl",
            epochs=1,
            val_every=100,
            learning_rate=config.learning_rate,
        ),
        "nextlat-warmup-sft-100": ArmSpec(
            name="nextlat-warmup-sft-100",
            train_size=100,
            train_file=config.splits_dir / "train-100.jsonl",
            warmup_file=warmup_file,
            warmup_epochs=1,
            warmup_lr=1e-4,
            zero_init_dynamics=True,
            lambda_mse=0.2,
            lambda_kl=0.1,
            epochs=1,
            val_every=5,
            learning_rate=config.learning_rate,
        ),
        "nextlat-warmup-sft-1000": ArmSpec(
            name="nextlat-warmup-sft-1000",
            train_size=1000,
            train_file=config.splits_dir / "train-1000.jsonl",
            warmup_file=warmup_file,
            warmup_epochs=1,
            warmup_lr=1e-4,
            zero_init_dynamics=True,
            lambda_mse=0.2,
            lambda_kl=0.1,
            epochs=1,
            val_every=25,
            learning_rate=config.learning_rate,
        ),
        "nextlat-cpt-sft-1000": ArmSpec(
            name="nextlat-cpt-sft-1000",
            train_size=1000,
            train_file=config.splits_dir / "train-1000.jsonl",
            warmup_file=warmup_file,
            warmup_epochs=1,
            warmup_lr=1e-4,
            cpt_epochs=1,
            cpt_lr=5e-6,
            zero_init_dynamics=True,
            lambda_mse=0.2,
            lambda_kl=0.1,
            epochs=1,
            val_every=25,
            learning_rate=config.learning_rate,
        ),
        "nextlat-warmup-sft-10000": ArmSpec(
            name="nextlat-warmup-sft-10000",
            train_size=10000,
            train_file=config.splits_dir / "train-10000.jsonl",
            warmup_file=warmup_file,
            warmup_epochs=1,
            warmup_lr=1e-4,
            zero_init_dynamics=True,
            lambda_mse=0.2,
            lambda_kl=0.1,
            epochs=1,
            val_every=100,
            learning_rate=config.learning_rate,
        ),
    }
    return [specs[name] for name in config.arms]


def load_api_key() -> str:
    key = os.environ.get("WANDB_API_KEY") or os.environ.get("CWSANDBOX_API_KEY")
    if key:
        return key.strip()
    key_path = Path.home() / ".config" / "forge-experiments" / "api-key"
    return key_path.read_text().strip()


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def load_task_scores(results_jsonl: Path) -> dict[str, dict[str, float]]:
    counts: dict[str, list[bool]] = {}
    for line in results_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        counts.setdefault(row["task_id"], []).append(bool(row["passed"]))
    scores: dict[str, dict[str, float]] = {}
    for task_id, outcomes in counts.items():
        n = len(outcomes)
        c = sum(outcomes)
        scores[task_id] = {
            "pass@1": pass_at_k(n, c, 1),
            "pass@10": pass_at_k(n, c, 10),
        }
    return scores


def paired_bootstrap_ci(
    base_results: Path,
    arm_results: Path,
    seed: int = 42,
    resamples: int = 10000,
) -> dict[str, Any]:
    base_map = load_task_scores(base_results)
    arm_map = load_task_scores(arm_results)
    tasks = sorted(set(base_map) & set(arm_map))
    if not tasks:
        raise ValueError("No overlapping tasks for paired bootstrap")
    rng = random.Random(seed)
    n = len(tasks)
    out: dict[str, Any] = {"tasks": n}
    for metric in ("pass@1", "pass@10"):
        diffs = [arm_map[t][metric] - base_map[t][metric] for t in tasks]
        point = sum(diffs) / n
        boot = []
        for _ in range(resamples):
            sample = [diffs[rng.randrange(n)] for _ in range(n)]
            boot.append(sum(sample) / n)
        boot.sort()
        out[metric] = {
            "delta": point,
            "ci95_low": boot[int(0.025 * resamples)],
            "ci95_high": boot[int(0.975 * resamples)],
        }
    return out


def write_file_with_retry(
    sb: Any, remote_path: str, data: bytes, retries: int = 3
) -> None:
    for attempt in range(retries):
        try:
            sb.write_file(remote_path, data).result()
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 * (attempt + 1))


def download_if_exists(sb: Any, remote_path: str, local_path: Path) -> bool:
    try:
        data = sb.read_file(remote_path, timeout_seconds=30).result()
    except Exception:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    return True


def run_arm_in_sandbox(
    arm: ArmSpec,
    config: AblationConfig,
    scripts_dir: Path,
    group_id: str,
) -> dict[str, Any]:
    import wandb
    from cwsandbox import AuthStrategy, Sandbox

    arm_dir = config.output_dir / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    resources = {
        "cpu": "4",
        "memory": "32Gi",
        "gpu": 1,
    }
    eff_lambda_mse = arm.lambda_mse if arm.lambda_mse is not None else config.lambda_mse
    eff_lambda_kl = arm.lambda_kl if arm.lambda_kl is not None else config.lambda_kl
    wb_run = None
    try:
        with Sandbox.run(
            auth=AuthStrategy.WANDB,
            placement_mode="serverless",
            container_image=CONTAINER_IMAGE,
            resources=resources,
            max_lifetime_seconds=config.max_lifetime_seconds,
        ) as sb:
            sb.exec(["bash", "-lc", "mkdir -p /tmp/work"]).result()
            write_file_with_retry(
                sb, "/tmp/work/train_lora.py", (scripts_dir / "train_lora.py").read_bytes()
            )
            write_file_with_retry(
                sb, "/tmp/work/nextlat.py", (scripts_dir / "nextlat.py").read_bytes()
            )
            write_file_with_retry(
                sb, "/tmp/work/evaluate.py", (scripts_dir / "evaluate.py").read_bytes()
            )
            write_file_with_retry(
                sb,
                "/tmp/work/val-100.jsonl",
                (config.splits_dir / "val-100.jsonl").read_bytes(),
            )
            write_file_with_retry(
                sb,
                "/tmp/work/test-100.jsonl",
                (config.splits_dir / "test-100.jsonl").read_bytes(),
            )
            if arm.train_file:
                write_file_with_retry(
                    sb, "/tmp/work/train.jsonl", arm.train_file.read_bytes()
                )
            if arm.warmup_file:
                write_file_with_retry(
                    sb, "/tmp/work/warmup.jsonl", arm.warmup_file.read_bytes()
                )

            extract_he = (
                "python -c \"import io,urllib.request,zipfile; "
                f"zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('{HUMAN_EVAL_ZIP}').read())).extractall('/tmp/work')\""
            )
            install_cmd = (
                "pip install -q transformers==4.56.2 peft==0.17.1 "
                "accelerate==1.10.1 bitsandbytes==0.48.1 pydantic==2.12.3 fire && "
                + extract_he
            )
            install_res = sb.exec(["bash", "-lc", install_cmd], timeout_seconds=600).result()
            if install_res.returncode != 0:
                raise RuntimeError(f"[{arm.name}] Dependency install failed: {install_res.stderr}")

            wb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                group=group_id,
                name=arm.name,
                config={
                    "arm": arm.name,
                    "mode": config.mode,
                    "objective": config.objective,
                    "lambda_mse": eff_lambda_mse,
                    "lambda_kl": eff_lambda_kl,
                    "proj_factor": config.proj_factor,
                    "zero_init_dynamics": arm.zero_init_dynamics,
                    "warmup_epochs": arm.warmup_epochs,
                    "warmup_lr": arm.warmup_lr,
                    "cpt_epochs": arm.cpt_epochs,
                    "cpt_lr": arm.cpt_lr,
                    "train_size": arm.train_size,
                    "epochs": arm.epochs,
                    "learning_rate": arm.learning_rate,
                    "accumulation": arm.accumulation,
                    "val_every": arm.val_every,
                    "seed": config.seed,
                    "model": "EleutherAI/gpt-j-6b",
                },
                reinit="create_new",
            )

            train_args = [
                "python -u /tmp/work/train_lora.py",
                f"--mode {config.mode}",
                f"--objective {config.objective}",
                f"--lambda-mse {eff_lambda_mse}",
                f"--lambda-kl {eff_lambda_kl}",
                f"--proj-factor {config.proj_factor}",
                f"--warmup-epochs {arm.warmup_epochs}",
                f"--warmup-lr {arm.warmup_lr}",
                f"--cpt-epochs {arm.cpt_epochs}",
                f"--cpt-lr {arm.cpt_lr}",
                "--val /tmp/work/val-100.jsonl",
                "--output /tmp/work/out",
                f"--epochs {arm.epochs}",
                f"--learning-rate {arm.learning_rate}",
                f"--accumulation {arm.accumulation}",
                f"--val-every {arm.val_every}",
                f"--seed {config.seed}",
                "--eval-benchmark /tmp/work/test-100.jsonl",
                "--eval-benchmark humaneval",
            ]
            if arm.zero_init_dynamics:
                train_args.append("--zero-init-dynamics")
            if arm.warmup_file:
                train_args.insert(2, "--warmup-train /tmp/work/warmup.jsonl")
            if arm.train_file:
                train_args.insert(2, "--train /tmp/work/train.jsonl")

            launch_cmd = (
                f"export PYTHONPATH=/tmp/work/human-eval-{HUMAN_EVAL_COMMIT}:/tmp/work && "
                + " ".join(train_args)
                + " 2>&1 | tee /tmp/work/stdout.log"
            )
            print(json.dumps({"arm": arm.name, "event": "started", "wandb_url": wb_run.url}), flush=True)

            logged_lines = 0

            def sync_metrics() -> None:
                nonlocal logged_lines
                try:
                    raw_metrics = sb.read_file(
                        "/tmp/work/out/metrics.jsonl", timeout_seconds=15
                    ).result()
                except Exception:
                    return
                lines = [
                    line
                    for line in raw_metrics.decode("utf-8", errors="replace").splitlines()
                    if line.strip()
                ]
                if len(lines) > logged_lines:
                    (arm_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")
                    for line in lines[logged_lines:]:
                        rec = json.loads(line)
                        step = rec["step"]
                        payload: dict[str, Any] = {
                            "step": step,
                            "train/examples_seen": rec.get("examples_seen", 0),
                            "train/epoch": rec.get("epoch", 0),
                        }
                        if "answer_token_nll" in rec:
                            payload["train/answer_token_nll"] = rec["answer_token_nll"]
                        if "val_answer_token_nll" in rec:
                            payload["val/answer_token_nll"] = rec["val_answer_token_nll"]
                        if "grad_norm" in rec:
                            payload["train/grad_norm"] = rec["grad_norm"]
                        if "learning_rate" in rec:
                            payload["train/learning_rate"] = rec["learning_rate"]
                        if "nextlat_mse_loss" in rec:
                            payload["train/nextlat_mse_loss"] = rec["nextlat_mse_loss"]
                        if "nextlat_kl_loss" in rec:
                            payload["train/nextlat_kl_loss"] = rec["nextlat_kl_loss"]
                        if "total_loss" in rec:
                            payload["train/total_loss"] = rec["total_loss"]
                        wb_run.log(payload, step=step)
                        print(json.dumps({"arm": arm.name, **rec}), flush=True)
                    logged_lines = len(lines)

            exec_timeout = max(600, config.max_lifetime_seconds - 300)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exec_pool:
                exec_future = exec_pool.submit(
                    lambda: sb.exec(
                        ["bash", "-lc", launch_cmd], timeout_seconds=exec_timeout
                    ).result()
                )
                while True:
                    try:
                        exec_res = exec_future.result(
                            timeout=config.poll_interval_seconds
                        )
                        break
                    except concurrent.futures.TimeoutError:
                        sync_metrics()

            sync_metrics()
            (arm_dir / "stdout.log").write_text(
                (exec_res.stdout or "") + "\n" + (exec_res.stderr or "")
            )
            download_if_exists(sb, "/tmp/work/out/config.json", arm_dir / "config.json")
            download_if_exists(sb, "/tmp/work/out/metrics.jsonl", arm_dir / "metrics.jsonl")
            for bench in ("eval_test_100", "eval_humaneval"):
                for fname in (
                    "summary.json",
                    "config.json",
                    "samples.jsonl",
                    "samples.jsonl_results.jsonl",
                    "raw.jsonl",
                ):
                    download_if_exists(
                        sb,
                        f"/tmp/work/out/{bench}/{fname}",
                        arm_dir / bench / fname,
                    )
            if not download_if_exists(
                sb, "/tmp/work/out/summary.json", arm_dir / "summary.json"
            ):
                tail = (arm_dir / "stdout.log").read_text()[-4000:]
                raise RuntimeError(
                    f"[{arm.name}] Process exited (code {exec_res.returncode}) without summary.json:\n{tail}"
                )

            summary = json.loads((arm_dir / "summary.json").read_text())
            summary["wandb_url"] = wb_run.url
            (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            for bench_name, scores in summary.get("evaluations", {}).items():
                for metric, val in scores.items():
                    wb_run.summary[f"{bench_name}/{metric}"] = val
            if summary.get("final_metrics", {}) and "val_answer_token_nll" in summary["final_metrics"]:
                wb_run.summary["val/final_answer_token_nll"] = summary["final_metrics"]["val_answer_token_nll"]
            print(
                json.dumps(
                    {
                        "arm": arm.name,
                        "event": "completed",
                        "evaluations": summary.get("evaluations"),
                        "wandb_url": wb_run.url,
                    }
                ),
                flush=True,
            )
            return summary
    finally:
        if wb_run is not None:
            wb_run.finish()


def run_ablation(config: AblationConfig) -> dict[str, Any]:
    api_key = load_api_key()
    os.environ["WANDB_API_KEY"] = api_key

    config.output_dir.mkdir(parents=True, exist_ok=True)
    arms = build_arms(config)
    scripts_dir = Path(__file__).resolve().parent
    group_id = f"gptj-{config.mode}-ablation-{int(time.time())}"

    results: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as pool:
        future_to_arm = {
            pool.submit(run_arm_in_sandbox, arm, config, scripts_dir, group_id): arm
            for arm in arms
        }
        for future in concurrent.futures.as_completed(future_to_arm):
            arm = future_to_arm[future]
            results[arm.name] = future.result()

    comparisons: dict[str, Any] = {}
    base_dir = config.output_dir / "base"
    if not (base_dir / "summary.json").exists():
        alt_base = config.output_dir.parent / "full-sft-ablation" / "base"
        if (alt_base / "summary.json").exists():
            base_dir = alt_base
    if base_dir.exists():
        for arm in arms:
            if arm.name == "base":
                continue
            arm_dir = config.output_dir / arm.name
            comparisons[arm.name] = {}
            for bench in ("eval_test_100", "eval_humaneval"):
                b_res = base_dir / bench / "samples.jsonl_results.jsonl"
                a_res = arm_dir / bench / "samples.jsonl_results.jsonl"
                if b_res.exists() and a_res.exists():
                    comparisons[arm.name][bench.removeprefix("eval_")] = (
                        paired_bootstrap_ci(b_res, a_res, seed=config.seed)
                    )

    summary_path = config.output_dir / "ablation_summary.json"
    existing_arms: dict[str, Any] = {}
    existing_comp: dict[str, Any] = {}
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text())
            existing_arms = prev.get("arms", {})
            existing_comp = prev.get("paired_bootstrap_vs_base", {})
        except Exception:
            pass
    existing_arms.update(results)
    existing_comp.update(comparisons)
    ablation_summary = {
        "group_id": group_id,
        "config": config.model_dump(mode="json"),
        "arms": existing_arms,
        "paired_bootstrap_vs_base": existing_comp,
    }
    summary_path.write_text(json.dumps(ablation_summary, indent=2))
    return ablation_summary


def main() -> None:
    config = parse_args()
    summary = run_ablation(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
