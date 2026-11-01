"""NextLat (Next-Latent Prediction) auxiliary dynamics head and loss for causal LMs.
Reference: JaydenTeoh/NextLat (commit 3770be6009cea2b3c455a9ce7f2ca88b504bb955).
At inference/evaluation time, the auxiliary dynamics head is discarded.
"""

from __future__ import annotations

from typing import Any


def build_nextlat_mask(
    ids: list[int], labels: list[int], eos_token_id: int
) -> list[bool]:
    """Return boolean mask over transitions t -> t+1 for t in 0 .. len(ids)-3."""
    if len(ids) != len(labels):
        raise ValueError("ids and labels must have equal length")
    if len(ids) < 3:
        return []
    return [
        labels[t + 1] != -100
        and labels[t + 2] != -100
        and ids[t + 1] != eos_token_id
        for t in range(len(ids) - 2)
    ]


def create_dynamics_model(
    n_embd: int,
    proj_factor: float = 0.5,
    dropout: float = 0.0,
    bias: bool = False,
    zero_init_output: bool = False,
) -> Any:
    """Create the residual MLP predicting h_{t+1} from (h_t, embed(x_{t+1}))."""
    import torch
    import torch.nn as nn

    class NextLatDynamicsModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_embd = n_embd
            input_dim = n_embd * 2
            hidden_dim = max(128, 128 * round((proj_factor * input_dim) / 128))
            self.hidden_state_dropout = (
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.norm_x = nn.LayerNorm(input_dim, bias=bias)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Linear(hidden_dim, n_embd, bias=bias),
            )
            self.apply(self._init_weights)
            if zero_init_output:
                last_linear = self.mlp[-1]
                assert isinstance(last_linear, nn.Linear)
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

        @staticmethod
        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(
            self, current_states: torch.Tensor, next_token_embeds: torch.Tensor
        ) -> torch.Tensor:
            hidden_states = self.hidden_state_dropout(current_states)
            x = torch.cat([next_token_embeds, hidden_states], dim=-1)
            x = self.norm_x(x)
            return current_states + self.mlp(x)

    return NextLatDynamicsModel()


def get_token_embeddings(model: Any, input_ids: Any) -> Any:
    return model.get_input_embeddings()(input_ids)


def get_lm_head(model: Any) -> Any:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        return output_embeddings
    return model.lm_head


def categorical_kl_loss(
    teacher_logits: Any,
    student_logits: Any,
    mask: Any,
) -> Any:
    """Compute masked KL(teacher || student) over vocabulary."""
    import torch.nn.functional as F

    log_q = F.log_softmax(teacher_logits.float(), dim=-1)
    log_p = F.log_softmax(student_logits.float(), dim=-1)
    kl_pointwise = F.kl_div(log_p, log_q, log_target=True, reduction="none")
    kl_per_token = kl_pointwise.sum(dim=-1)
    m = mask.to(dtype=kl_per_token.dtype)
    return (kl_per_token * m).sum() / m.sum().clamp_min(1.0)


def _compute_rollout_latent_losses(
    hidden_states: Any,
    logits: Any,
    token_embeds: Any,
    input_ids: Any,
    labels: Any,
    dynamics_model: Any,
    head_weight: Any,
    head_bias: Any,
    eos_token_id: int,
    rollout_steps: int = 1,
) -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as F

    seq_len = input_ids.size(1)
    max_r = min(max(1, rollout_steps), seq_len - 2)
    curr_h = hidden_states[:, :-2, :]
    cum_mask = (
        (labels[:, 1:-1] != -100)
        & (labels[:, 2:] != -100)
        & (input_ids[:, 1:-1] != eos_token_id)
    )

    total_mse = torch.zeros((), device=input_ids.device, dtype=torch.float32)
    total_kl = torch.zeros((), device=input_ids.device, dtype=torch.float32)
    total_weight = 0.0
    first_pairs = cum_mask.sum()

    for r in range(1, max_r + 1):
        if r > 1:
            curr_h = pred_h[:, :-1, :]
            step_mask = (
                (labels[:, r:-1] != -100)
                & (labels[:, r + 1 :] != -100)
                & (input_ids[:, r:-1] != eos_token_id)
            )
            cum_mask = cum_mask[:, :-1] & step_mask

        next_emb = token_embeds[:, r:-1, :]
        target_h = hidden_states[:, r:-1, :].detach()
        teacher_logits = logits[:, r:-1, :].detach()

        pred_h = dynamics_model(curr_h, next_emb)

        mse_elem = F.smooth_l1_loss(pred_h.float(), target_h.float(), reduction="none")
        w = cum_mask.unsqueeze(-1).to(dtype=mse_elem.dtype)
        denom = (w.sum() * mse_elem.size(-1)).clamp_min(1.0)
        step_mse = (mse_elem * w).sum() / denom

        student_logits = F.linear(pred_h, head_weight, head_bias)
        step_kl = categorical_kl_loss(teacher_logits, student_logits, cum_mask)

        step_w = 0.5 ** (r - 1)
        total_mse = total_mse + step_w * step_mse
        total_kl = total_kl + step_w * step_kl
        total_weight += step_w

    return total_mse / total_weight, total_kl / total_weight, first_pairs


def compute_nextlat_losses(
    model: Any,
    dynamics_model: Any,
    input_ids: Any,
    labels: Any,
    eos_token_id: int,
    rollout_steps: int = 1,
) -> dict[str, Any]:
    """Run forward pass and compute NTP, NextLat Smooth-L1 (MSE), and KL losses."""
    import torch

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        output_hidden_states=True,
    )
    ntp_loss = outputs.loss
    hidden_states = outputs.hidden_states[-1]
    token_embeds = get_token_embeddings(model, input_ids)

    if input_ids.size(1) < 3:
        zero = torch.zeros_like(ntp_loss)
        return {
            "ntp_loss": ntp_loss,
            "mse_loss": zero,
            "kl_loss": zero,
            "latent_pairs": torch.tensor(0, device=input_ids.device),
        }

    lm_head = get_lm_head(model)
    head_weight = lm_head.weight.detach()
    head_bias = lm_head.bias.detach() if lm_head.bias is not None else None

    mse_loss, kl_loss, latent_pairs = _compute_rollout_latent_losses(
        hidden_states=hidden_states,
        logits=outputs.logits,
        token_embeds=token_embeds,
        input_ids=input_ids,
        labels=labels,
        dynamics_model=dynamics_model,
        head_weight=head_weight,
        head_bias=head_bias,
        eos_token_id=eos_token_id,
        rollout_steps=rollout_steps,
    )

    return {
        "ntp_loss": ntp_loss,
        "mse_loss": mse_loss,
        "kl_loss": kl_loss,
        "latent_pairs": latent_pairs,
    }


def compute_dynamics_warmup_losses(
    model: Any,
    dynamics_model: Any,
    input_ids: Any,
    labels: Any,
    eos_token_id: int,
    rollout_steps: int = 1,
) -> dict[str, Any]:
    """Compute NextLat Smooth-L1 and KL losses with the backbone frozen under no_grad."""
    import torch

    if input_ids.size(1) < 3:
        zero = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        return {
            "mse_loss": zero,
            "kl_loss": zero,
            "latent_pairs": torch.tensor(0, device=input_ids.device),
        }

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1].detach()
        logits = outputs.logits.detach()
        token_embeds = get_token_embeddings(model, input_ids).detach()
        lm_head = get_lm_head(model)
        head_weight = lm_head.weight.detach()
        head_bias = lm_head.bias.detach() if lm_head.bias is not None else None

    mse_loss, kl_loss, latent_pairs = _compute_rollout_latent_losses(
        hidden_states=hidden_states,
        logits=logits,
        token_embeds=token_embeds,
        input_ids=input_ids,
        labels=labels,
        dynamics_model=dynamics_model,
        head_weight=head_weight,
        head_bias=head_bias,
        eos_token_id=eos_token_id,
        rollout_steps=rollout_steps,
    )

    return {
        "mse_loss": mse_loss,
        "kl_loss": kl_loss,
        "latent_pairs": latent_pairs,
    }


def crop_past_key_values(past_key_values: Any, max_length: int) -> Any:
    """Crop KV cache to max_length tokens across DynamicCache or legacy tuple formats."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(max_length)
        return past_key_values
    if isinstance(past_key_values, (tuple, list)):
        return tuple(
            tuple(t[:, :, :max_length, :] for t in layer) for layer in past_key_values
        )
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def greedy_generate(
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    eos_token_id: int,
) -> dict[str, Any]:
    """Standard greedy autoregressive decoding with KV caching."""
    import time
    import torch

    if max_new_tokens <= 0:
        return {"tokens": [], "target_forwards": 0, "elapsed_seconds": 0.0}

    device = input_ids.device
    generated: list[int] = []
    target_forwards = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        target_forwards += 1
        past_key_values = out.past_key_values
        next_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_tok)

        while len(generated) < max_new_tokens and next_tok != eos_token_id:
            cur_in = torch.tensor([[next_tok]], device=device, dtype=input_ids.dtype)
            out = model(
                input_ids=cur_in, past_key_values=past_key_values, use_cache=True
            )
            target_forwards += 1
            past_key_values = out.past_key_values
            next_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_tok)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "tokens": generated,
        "target_forwards": target_forwards,
        "elapsed_seconds": elapsed,
    }


def speculative_generate(
    model: Any,
    dynamics_model: Any,
    input_ids: Any,
    max_new_tokens: int,
    draft_k: int,
    eos_token_id: int,
) -> dict[str, Any]:
    """Lossless greedy self-speculative decoding using NextLatDynamicsModel as drafter."""
    import time
    import torch
    import torch.nn.functional as F

    if max_new_tokens <= 0:
        return {
            "tokens": [],
            "target_forwards": 0,
            "draft_steps": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "accepted_by_pos": [0] * max(1, draft_k),
            "proposed_by_pos": [0] * max(1, draft_k),
            "elapsed_seconds": 0.0,
        }
    if draft_k <= 0:
        res = greedy_generate(model, input_ids, max_new_tokens, eos_token_id)
        return {
            **res,
            "draft_steps": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "accepted_by_pos": [],
            "proposed_by_pos": [],
        }

    device = input_ids.device
    generated: list[int] = []
    target_forwards = 0
    draft_steps = 0
    draft_proposed = 0
    draft_accepted = 0
    accepted_by_pos = [0] * draft_k
    proposed_by_pos = [0] * draft_k

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        lm_head = get_lm_head(model)
        head_weight = lm_head.weight
        head_bias = lm_head.bias if lm_head.bias is not None else None

        out = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        target_forwards += 1
        past_key_values = out.past_key_values
        cache_len = input_ids.size(1)
        curr_h = out.hidden_states[-1][:, -1:, :]
        curr_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(curr_tok)

        while len(generated) < max_new_tokens and curr_tok != eos_token_id:
            remaining = max_new_tokens - len(generated)
            k_step = min(draft_k, remaining)
            draft_tokens: list[int] = []
            sim_h = curr_h
            sim_tok = curr_tok

            for j in range(k_step):
                tok_tensor = torch.tensor(
                    [[sim_tok]], device=device, dtype=input_ids.dtype
                )
                tok_emb = embed_layer(tok_tensor)
                sim_h = dynamics_model(sim_h, tok_emb)
                logits_j = F.linear(sim_h, head_weight, head_bias)
                sim_tok = int(logits_j[:, -1, :].argmax(dim=-1).item())
                draft_tokens.append(sim_tok)
                proposed_by_pos[j] += 1
                draft_proposed += 1
                if sim_tok == eos_token_id:
                    break

            actual_k = len(draft_tokens)
            verify_ids = torch.tensor(
                [[curr_tok, *draft_tokens]], device=device, dtype=input_ids.dtype
            )
            v_out = model(
                input_ids=verify_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            target_forwards += 1
            draft_steps += 1
            past_key_values = v_out.past_key_values
            v_logits = v_out.logits
            v_hidden = v_out.hidden_states[-1]

            n_acc = 0
            hit_eos = False
            for j in range(actual_k):
                target_tok = int(v_logits[:, j, :].argmax(dim=-1).item())
                if draft_tokens[j] == target_tok:
                    n_acc += 1
                    draft_accepted += 1
                    accepted_by_pos[j] += 1
                    generated.append(target_tok)
                    if target_tok == eos_token_id or len(generated) >= max_new_tokens:
                        hit_eos = True
                        break
                else:
                    break

            if hit_eos or len(generated) >= max_new_tokens:
                break

            bonus_tok = int(v_logits[:, n_acc, :].argmax(dim=-1).item())
            generated.append(bonus_tok)
            curr_tok = bonus_tok
            curr_h = v_hidden[:, n_acc : n_acc + 1, :]
            cache_len = cache_len + n_acc + 1
            past_key_values = crop_past_key_values(past_key_values, cache_len)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    return {
        "tokens": generated[:max_new_tokens],
        "target_forwards": target_forwards,
        "draft_steps": draft_steps,
        "draft_tokens_proposed": draft_proposed,
        "draft_tokens_accepted": draft_accepted,
        "accepted_by_pos": accepted_by_pos,
        "proposed_by_pos": proposed_by_pos,
        "elapsed_seconds": elapsed,
    }


def truncate_at_stops(text: str, stops: tuple[str, ...] = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")) -> str:
    """Truncate generated completion at the first occurrence of any stop string."""
    indices = [text.find(s) for s in stops if s in text]
    return text[: min(indices)] if indices else text


def run_speculative_decoding_benchmark(
    model: Any,
    dynamics_model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 128,
    draft_ks: tuple[int, ...] = (1, 2, 3, 4),
    task_ids: list[str] | None = None,
    problem_file: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Benchmark greedy autoregressive vs NextLat speculative decoding across draft_ks."""
    import json
    import torch

    was_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    model.eval()
    dynamics_model.eval()
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    if prompts:
        warm_ids = torch.tensor(
            [tokenizer(prompts[0], add_special_tokens=False)["input_ids"]],
            device=device,
        )
        greedy_generate(model, warm_ids, min(16, max_new_tokens), eos_id)
        speculative_generate(
            model, dynamics_model, warm_ids, min(16, max_new_tokens), 2, eos_id
        )

    baseline_tokens: list[list[int]] = []
    base_total_tokens = 0
    base_total_forwards = 0
    base_total_sec = 0.0

    for prompt in prompts:
        ids = torch.tensor(
            [tokenizer(prompt, add_special_tokens=False)["input_ids"]],
            device=device,
        )
        res = greedy_generate(model, ids, max_new_tokens, eos_id)
        baseline_tokens.append(res["tokens"])
        base_total_tokens += len(res["tokens"])
        base_total_forwards += res["target_forwards"]
        base_total_sec += res["elapsed_seconds"]

    base_tps = base_total_tokens / max(base_total_sec, 1e-9)
    greedy_summary: dict[str, Any] = {
        "total_tokens": base_total_tokens,
        "target_forwards": base_total_forwards,
        "elapsed_seconds": base_total_sec,
        "tokens_per_sec": base_tps,
        "tokens_per_target_forward": base_total_tokens / max(base_total_forwards, 1),
    }

    if task_ids is not None and problem_file is not None and output_dir is not None:
        from human_eval.evaluation import evaluate_functional_correctness

        output_dir.mkdir(parents=True, exist_ok=True)
        greedy_samples_path = output_dir / "greedy_samples.jsonl"
        with greedy_samples_path.open("w") as f:
            for tid, toks in zip(task_ids, baseline_tokens):
                raw_text = tokenizer.decode(toks, skip_special_tokens=True)
                f.write(
                    json.dumps(
                        {"task_id": tid, "completion": truncate_at_stops(raw_text)}
                    )
                    + "\n"
                )
        greedy_scores = evaluate_functional_correctness(
            str(greedy_samples_path),
            k=[1],
            n_workers=4,
            timeout=3,
            problem_file=str(problem_file),
        )
        greedy_summary["pass@1"] = float(greedy_scores["pass@1"])

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
            res = speculative_generate(
                model, dynamics_model, ids, max_new_tokens, k, eos_id
            )
            spec_tokens_list.append(res["tokens"])
            if res["tokens"] == baseline_tokens[idx]:
                exact_matches += 1
            total_tokens += len(res["tokens"])
            total_forwards += res["target_forwards"]
            total_steps += res["draft_steps"]
            total_proposed += res["draft_tokens_proposed"]
            total_accepted += res["draft_tokens_accepted"]
            total_sec += res["elapsed_seconds"]
            for j in range(k):
                acc_by_pos[j] += res["accepted_by_pos"][j]
                prop_by_pos[j] += res["proposed_by_pos"][j]

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
            "forward_pass_reduction_factor": base_total_forwards / max(total_forwards, 1),
            "acceptance_rate_by_pos": [
                acc_by_pos[j] / max(prop_by_pos[j], 1) for j in range(k)
            ],
            "elapsed_seconds": total_sec,
            "tokens_per_sec": tps,
            "speedup_vs_greedy": tps / max(base_tps, 1e-9),
            "exact_token_match_rate": exact_matches / max(len(prompts), 1),
        }
        if task_ids is not None and problem_file is not None and output_dir is not None:
            from human_eval.evaluation import evaluate_functional_correctness

            k_samples_path = output_dir / f"spec_k{k}_samples.jsonl"
            with k_samples_path.open("w") as f:
                for tid, toks in zip(task_ids, spec_tokens_list):
                    raw_text = tokenizer.decode(toks, skip_special_tokens=True)
                    f.write(
                        json.dumps(
                            {"task_id": tid, "completion": truncate_at_stops(raw_text)}
                        )
                        + "\n"
                    )
            k_scores = evaluate_functional_correctness(
                str(k_samples_path),
                k=[1],
                n_workers=4,
                timeout=3,
                problem_file=str(problem_file),
            )
            k_entry["pass@1"] = float(k_scores["pass@1"])
        summary["speculative"][f"k={k}"] = k_entry

    model.config.use_cache = was_cache
    return summary
