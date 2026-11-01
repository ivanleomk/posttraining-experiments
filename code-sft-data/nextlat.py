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


def compute_nextlat_losses(
    model: Any,
    dynamics_model: Any,
    input_ids: Any,
    labels: Any,
    eos_token_id: int,
) -> dict[str, Any]:
    """Run forward pass and compute NTP, NextLat Smooth-L1 (MSE), and KL losses."""
    import torch
    import torch.nn.functional as F

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

    curr_h = hidden_states[:, :-2, :]
    target_h = hidden_states[:, 1:-1, :].detach()
    next_emb = token_embeds[:, 1:-1, :]
    teacher_logits = outputs.logits[:, 1:-1, :].detach()

    valid_mask = (
        (labels[:, 1:-1] != -100)
        & (labels[:, 2:] != -100)
        & (input_ids[:, 1:-1] != eos_token_id)
    )

    pred_h = dynamics_model(curr_h, next_emb)

    mse_elem = F.smooth_l1_loss(pred_h.float(), target_h.float(), reduction="none")
    w = valid_mask.unsqueeze(-1).to(dtype=mse_elem.dtype)
    denom = (w.sum() * mse_elem.size(-1)).clamp_min(1.0)
    mse_loss = (mse_elem * w).sum() / denom

    lm_head = get_lm_head(model)
    head_weight = lm_head.weight.detach()
    head_bias = lm_head.bias.detach() if lm_head.bias is not None else None
    student_logits = F.linear(pred_h, head_weight, head_bias)
    kl_loss = categorical_kl_loss(teacher_logits, student_logits, valid_mask)

    return {
        "ntp_loss": ntp_loss,
        "mse_loss": mse_loss,
        "kl_loss": kl_loss,
        "latent_pairs": valid_mask.sum(),
    }


def compute_dynamics_warmup_losses(
    model: Any,
    dynamics_model: Any,
    input_ids: Any,
    labels: Any,
    eos_token_id: int,
) -> dict[str, Any]:
    """Compute NextLat Smooth-L1 and KL losses with the backbone frozen under no_grad."""
    import torch
    import torch.nn.functional as F

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
        token_embeds = get_token_embeddings(model, input_ids).detach()
        teacher_logits = outputs.logits[:, 1:-1, :].detach()
        lm_head = get_lm_head(model)
        head_weight = lm_head.weight.detach()
        head_bias = lm_head.bias.detach() if lm_head.bias is not None else None

    curr_h = hidden_states[:, :-2, :]
    target_h = hidden_states[:, 1:-1, :]
    next_emb = token_embeds[:, 1:-1, :]

    valid_mask = (
        (labels[:, 1:-1] != -100)
        & (labels[:, 2:] != -100)
        & (input_ids[:, 1:-1] != eos_token_id)
    )

    pred_h = dynamics_model(curr_h, next_emb)

    mse_elem = F.smooth_l1_loss(pred_h.float(), target_h.float(), reduction="none")
    w = valid_mask.unsqueeze(-1).to(dtype=mse_elem.dtype)
    denom = (w.sum() * mse_elem.size(-1)).clamp_min(1.0)
    mse_loss = (mse_elem * w).sum() / denom

    student_logits = F.linear(pred_h, head_weight, head_bias)
    kl_loss = categorical_kl_loss(teacher_logits, student_logits, valid_mask)

    return {
        "mse_loss": mse_loss,
        "kl_loss": kl_loss,
        "latent_pairs": valid_mask.sum(),
    }
