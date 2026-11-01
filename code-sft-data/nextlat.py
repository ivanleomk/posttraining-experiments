"""First-class NextLat subclass of HuggingFace GPTJForCausalLM (`GPTJNextLatForCausalLM`)."""

from __future__ import annotations

import time
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPTJConfig, GPTJForCausalLM
    from transformers.generation.candidate_generator import CandidateGenerator
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = type("nn", (), {"Module": object})  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    GPTJConfig = object  # type: ignore[misc,assignment]
    GPTJForCausalLM = object  # type: ignore[misc,assignment]
    CandidateGenerator = object  # type: ignore[misc,assignment]


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


class NextLatDynamicsModel(nn.Module):
    """Residual MLP predicting h_{t+1} from (h_t, embed(x_{t+1}))."""

    def __init__(
        self,
        n_embd: int,
        proj_factor: float = 0.5,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        input_dim = n_embd * 2
        hidden_dim = max(128, 128 * round((proj_factor * input_dim) / 128))
        self.norm_x = nn.LayerNorm(input_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd, bias=False),
        )
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        if zero_init_output:
            nn.init.zeros_(self.mlp[-1].weight)

    def forward(
        self, current_states: torch.Tensor, next_token_embeds: torch.Tensor
    ) -> torch.Tensor:
        x = self.norm_x(torch.cat([next_token_embeds, current_states], dim=-1))
        return current_states + self.mlp(x)


class NextLatCandidateGenerator(CandidateGenerator):
    """HuggingFace CandidateGenerator using GPTJNextLatForCausalLM.nextlat_dynamics."""

    def __init__(self, model: GPTJNextLatForCausalLM, draft_k: int) -> None:
        self.model = model
        self.num_assistant_tokens = draft_k
        self.target_forwards = 0
        self.draft_steps = 0
        self.draft_tokens_proposed = 0
        self.draft_tokens_accepted = 0
        self.accepted_by_pos = [0] * max(1, draft_k)
        self.proposed_by_pos = [0] * max(1, draft_k)
        self.verified_hidden: torch.Tensor | None = None
        self.last_proposed: list[int] = []

    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.FloatTensor | None]:
        if self.verified_hidden is None or self.num_assistant_tokens <= 0:
            self.last_proposed = []
            return input_ids, None

        sim_h = self.verified_hidden
        sim_tok = int(input_ids[0, -1].item())
        eos_id = self.model.config.eos_token_id
        draft_tokens: list[int] = []
        draft_logits: list[torch.Tensor] = []

        with torch.no_grad():
            for j in range(self.num_assistant_tokens):
                tok_t = torch.tensor([[sim_tok]], device=input_ids.device)
                tok_emb = self.model.transformer.wte(tok_t)
                sim_h = self.model.nextlat_dynamics(sim_h, tok_emb)
                logits_j = self.model.lm_head(sim_h)
                sim_tok = int(logits_j[:, -1, :].argmax(dim=-1).item())
                draft_tokens.append(sim_tok)
                draft_logits.append(logits_j[:, -1:, :].float())
                self.proposed_by_pos[j] += 1
                self.draft_tokens_proposed += 1
                if sim_tok == eos_id:
                    break

        self.draft_steps += 1
        self.last_proposed = draft_tokens
        if not draft_tokens:
            return input_ids, None
        cand_ids = torch.cat(
            [input_ids, torch.tensor([draft_tokens], device=input_ids.device)], dim=-1
        )
        return cand_ids, torch.cat(draft_logits, dim=1)

    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ) -> None:
        n_acc = int(num_matches)
        latest_h = self.model._last_hidden_state
        if latest_h is not None:
            base_idx = latest_h.size(1) - len(self.last_proposed) - 1
            self.verified_hidden = latest_h[:, base_idx + n_acc : base_idx + n_acc + 1, :]
        if self.last_proposed:
            self.draft_tokens_accepted += n_acc
            for j in range(min(n_acc, len(self.last_proposed))):
                self.accepted_by_pos[j] += 1


class GPTJNextLatConfig(GPTJConfig):
    def __init__(
        self,
        nextlat_proj_factor: float = 0.5,
        nextlat_zero_init: bool = True,
        nextlat_lambda_mse: float = 0.2,
        nextlat_lambda_kl: float = 0.1,
        nextlat_rollout_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.nextlat_proj_factor = nextlat_proj_factor
        self.nextlat_zero_init = nextlat_zero_init
        self.nextlat_lambda_mse = nextlat_lambda_mse
        self.nextlat_lambda_kl = nextlat_lambda_kl
        self.nextlat_rollout_steps = nextlat_rollout_steps


class GPTJNextLatForCausalLM(GPTJForCausalLM):
    """First-class GPTJForCausalLM subclass with built-in NextLat training and HF speculative decoding."""

    config_class = GPTJNextLatConfig

    def __init__(self, config: GPTJConfig) -> None:
        super().__init__(config)
        self.nextlat_dynamics = NextLatDynamicsModel(
            n_embd=config.n_embd,
            proj_factor=getattr(config, "nextlat_proj_factor", 0.5),
            zero_init_output=getattr(config, "nextlat_zero_init", True),
        )
        self.warmup_only = False
        self._last_hidden_state: torch.Tensor | None = None
        self._active_candidate_gen: NextLatCandidateGenerator | None = None
        self.last_gen_stats: dict[str, Any] = {}

    def _compute_rollout_losses(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = input_ids.size(1)
        if seq_len < 3:
            z = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            return z, z

        eos_id = self.config.eos_token_id
        rollout_steps = getattr(self.config, "nextlat_rollout_steps", 1)
        max_r = min(max(1, rollout_steps), seq_len - 2)
        token_embeds = self.transformer.wte(input_ids)
        head_w = self.lm_head.weight.detach()
        head_b = self.lm_head.bias.detach() if self.lm_head.bias is not None else None

        curr_h = hidden_states[:, :-2, :]
        cum_mask = (
            (labels[:, 1:-1] != -100)
            & (labels[:, 2:] != -100)
            & (input_ids[:, 1:-1] != eos_id)
        )
        total_mse = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        total_kl = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        total_weight = 0.0

        for r in range(1, max_r + 1):
            if r > 1:
                curr_h = pred_h[:, :-1, :]
                step_mask = (
                    (labels[:, r:-1] != -100)
                    & (labels[:, r + 1 :] != -100)
                    & (input_ids[:, r:-1] != eos_id)
                )
                cum_mask = cum_mask[:, :-1] & step_mask

            pred_h = self.nextlat_dynamics(curr_h, token_embeds[:, r:-1, :])
            target_h = hidden_states[:, r:-1, :].detach()
            teacher_logits = logits[:, r:-1, :].detach()

            mse_elem = F.smooth_l1_loss(pred_h.float(), target_h.float(), reduction="none")
            w = cum_mask.unsqueeze(-1).to(dtype=mse_elem.dtype)
            step_mse = (mse_elem * w).sum() / (w.sum() * mse_elem.size(-1)).clamp_min(1.0)

            log_q = F.log_softmax(teacher_logits.float(), dim=-1)
            log_p = F.log_softmax(F.linear(pred_h, head_w, head_b).float(), dim=-1)
            kl_tok = F.kl_div(log_p, log_q, log_target=True, reduction="none").sum(dim=-1)
            m = cum_mask.to(dtype=kl_tok.dtype)
            step_kl = (kl_tok * m).sum() / m.sum().clamp_min(1.0)

            step_w = 0.5 ** (r - 1)
            total_mse = total_mse + step_w * step_mse
            total_kl = total_kl + step_w * step_kl
            total_weight += step_w

        return total_mse / total_weight, total_kl / total_weight

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        *args: Any,
        labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._active_candidate_gen is not None:
            kwargs["output_hidden_states"] = True
            out = super().forward(input_ids=input_ids, *args, labels=labels, **kwargs)
            self._last_hidden_state = out.hidden_states[-1].detach()
            self._active_candidate_gen.target_forwards += 1
            return out

        if labels is None or (not self.training and not self.warmup_only):
            return super().forward(input_ids=input_ids, *args, labels=labels, **kwargs)

        if self.warmup_only:
            with torch.no_grad():
                out = super().forward(
                    input_ids=input_ids, *args, output_hidden_states=True, **kwargs
                )
            mse_loss, kl_loss = self._compute_rollout_losses(
                out.hidden_states[-1].detach(), out.logits.detach(), input_ids, labels
            )
            out.loss = mse_loss + kl_loss
            out["mse_loss"] = mse_loss
            out["kl_loss"] = kl_loss
            return out

        out = super().forward(
            input_ids=input_ids,
            *args,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )
        ntp_loss = out.loss
        mse_loss, kl_loss = self._compute_rollout_losses(
            out.hidden_states[-1], out.logits, input_ids, labels
        )
        l_mse = getattr(self.config, "nextlat_lambda_mse", 0.2)
        l_kl = getattr(self.config, "nextlat_lambda_kl", 0.1)
        out.loss = ntp_loss + l_mse * mse_loss + l_kl * kl_loss
        out["ntp_loss"] = ntp_loss
        out["mse_loss"] = mse_loss
        out["kl_loss"] = kl_loss
        return out

    def _get_candidate_generator(self, **kwargs: Any) -> CandidateGenerator:
        if self._active_candidate_gen is not None:
            return self._active_candidate_gen
        return super()._get_candidate_generator(**kwargs)

    def generate(
        self,
        *args: Any,
        num_assistant_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        if not num_assistant_tokens or num_assistant_tokens <= 0:
            return super().generate(*args, **kwargs)

        inp = kwargs.get("input_ids", args[0] if args else None)
        if inp is not None and "attention_mask" not in kwargs:
            kwargs["attention_mask"] = torch.ones_like(inp)
        kwargs.setdefault("pad_token_id", self.config.eos_token_id)
        kwargs.setdefault("eos_token_id", self.config.eos_token_id)
        kwargs["prompt_lookup_num_tokens"] = num_assistant_tokens

        cand_gen = NextLatCandidateGenerator(self, draft_k=num_assistant_tokens)
        self._active_candidate_gen = cand_gen
        dev = inp.device if inp is not None else self.device
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out = super().generate(*args, **kwargs)
        finally:
            self._active_candidate_gen = None
        if dev.type == "cuda":
            torch.cuda.synchronize()
        self.last_gen_stats = {
            "target_forwards": cand_gen.target_forwards,
            "draft_steps": cand_gen.draft_steps,
            "draft_tokens_proposed": cand_gen.draft_tokens_proposed,
            "draft_tokens_accepted": cand_gen.draft_tokens_accepted,
            "accepted_by_pos": cand_gen.accepted_by_pos,
            "proposed_by_pos": cand_gen.proposed_by_pos,
            "elapsed_seconds": time.perf_counter() - t0,
        }
        return out
