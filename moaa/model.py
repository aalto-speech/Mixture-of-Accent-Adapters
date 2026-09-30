"""
MoAA model: frozen Whisper + multi-task heads + accent-conditioned mixture of adapters.

Pipeline (per utterance):
  1. Whisper encoder -> hidden states H  [B, T, D]
  2. Mean-pool H -> pooled               [B, D]   (optionally through a linear projection)
  3. Heads on pooled:
       - accented-or-not (2 classes)
       - accent ID (num_accents classes)
       - gender (2 classes) behind a Gradient Reversal Layer (adversarial)
  4. Soft gate p = P(accented)
  5. Accent conditioning vector e = p * (softmax(accent) @ accent_embed) + (1 - p) * neutral_embed
  6. H_cond = cond_proj(H + e)
  7. Router mixes N bottleneck adapters on H_cond -> A
  8. Encoder output for decoder = p * A + (1 - p) * H
  9. Whisper decoder -> ASR logits
"""

import json
from pathlib import Path
from typing import Dict, Optional

import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


# =============================================================================
# Adapter + Router
# =============================================================================
class Adapter(nn.Module):
    """Simple bottleneck adapter with residual connection."""

    def __init__(self, hidden_size, bottleneck_size=256):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class AdapterRouter(nn.Module):
    """Router learns to mix multiple adapters dynamically (utterance-level soft routing)."""

    def __init__(self, hidden_size, num_adapters):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, num_adapters),
            nn.Softmax(dim=-1),
        )

    def forward(self, hidden_states, adapters: nn.ModuleDict):
        B, T, H = hidden_states.shape
        pooled = hidden_states.mean(dim=1)      # [B,H]
        weights = self.gate(pooled)             # [B,A]

        combined = 0
        for i, key in enumerate(adapters.keys()):
            adapted = adapters[key](hidden_states)   # [B,T,H]
            w = weights[:, i].view(B, 1, 1)
            combined = combined + w * adapted
        return combined, weights


# =============================================================================
# Gradient Reversal Layer (GRL)
# =============================================================================
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return _GRL.apply(x, self.lambd)


# =============================================================================
# Unfreeze last N Whisper layers (used by the `unfreeze_last2` variant)
# =============================================================================
def unfreeze_last_n_whisper_layers(whisper_model: WhisperForConditionalGeneration, n: int, logger=None):
    """
    Unfreeze last n transformer blocks from BOTH encoder and decoder.
    Keeps embeddings / early layers frozen. Also unfreezes the final LayerNorms.
    """
    n = int(n)
    if n <= 0:
        return

    enc = whisper_model.model.encoder
    dec = whisper_model.model.decoder

    enc_layers = getattr(enc, "layers", None)
    dec_layers = getattr(dec, "layers", None)

    if enc_layers is None or dec_layers is None:
        raise RuntimeError("Could not find encoder/decoder .layers in Whisper model. Check HF Transformers version.")

    n_enc = min(n, len(enc_layers))
    n_dec = min(n, len(dec_layers))

    for layer in enc_layers[-n_enc:]:
        for p in layer.parameters():
            p.requires_grad = True

    for layer in dec_layers[-n_dec:]:
        for p in layer.parameters():
            p.requires_grad = True

    if hasattr(enc, "layer_norm"):
        for p in enc.layer_norm.parameters():
            p.requires_grad = True
    if hasattr(dec, "layer_norm"):
        for p in dec.layer_norm.parameters():
            p.requires_grad = True

    if logger is not None:
        logger.info("Unfroze last %d encoder layers and last %d decoder layers.", n_enc, n_dec)


# =============================================================================
# Whisper + Multi-task Heads + Accent-conditioned Adapters
# =============================================================================
class WhisperAccentedConditionalAdapters(nn.Module):
    def __init__(
        self,
        whisper_model: WhisperForConditionalGeneration,
        num_adapters: int = 6,
        bottleneck_size: int = 192,
        num_accents: int = 10,
        accent_ignore_index: int = -100,
        loss_weights=(1.0, 1.0, 1.0, 1.0),
        freeze_whisper: bool = True,
        grl_lambda: float = 1.0,
        use_linear_proj: bool = False,
        unfreeze_last_n: int = 0,
        proj_in_generate: bool = False,
    ):
        """
        Args:
            loss_weights: (w_asr, w_accented_or_not, w_accent, w_gender).
            use_linear_proj: apply a D->D linear projection to the pooled encoder
                representation before the classification heads (`linear_proj` variant).
            unfreeze_last_n: when freeze_whisper=True, re-enable gradients for the last N
                encoder and decoder blocks (`unfreeze_last2` variant uses N=2).
            proj_in_generate: also apply `linear_proj` inside `generate()`.
                The released checkpoints were trained AND evaluated with this set to False
                (the projection is only used in `forward()` / `predict_clf()`), so keep the
                default to reproduce the reported numbers.
        """
        super().__init__()
        self.whisper = whisper_model

        H = whisper_model.model.encoder.config.d_model
        self.num_accents = int(num_accents)
        self.accent_ignore_index = int(accent_ignore_index)
        self.use_linear_proj = bool(use_linear_proj)
        self.proj_in_generate = bool(proj_in_generate)

        self.w_asr, self.w_accbin, self.w_accent, self.w_gender = loss_weights

        # Linear projection on pooled features (only in `linear_proj` variant)
        if self.use_linear_proj:
            self.linear_proj = nn.Linear(H, H)

        # Heads
        self.head_accented = nn.Linear(H, 2)
        self.head_accent = nn.Linear(H, self.num_accents)

        self.grl_lambda = float(grl_lambda)
        self.grl = GradientReversal(lambd=self.grl_lambda)
        self.head_gender = nn.Linear(H, 2)

        # Accent prototypes + neutral
        self.accent_embed = nn.Embedding(self.num_accents, H)
        self.neutral_embed = nn.Parameter(torch.zeros(H))

        # Conditioning projection
        self.cond_proj = nn.Linear(H, H)

        # Adapters + router
        self.adapters = nn.ModuleDict({
            f"adapter_{i}": Adapter(H, bottleneck_size) for i in range(num_adapters)
        })
        self.router = AdapterRouter(H, num_adapters)

        if freeze_whisper:
            for p in self.whisper.model.encoder.parameters():
                p.requires_grad = False
            for p in self.whisper.model.decoder.parameters():
                p.requires_grad = False

            # Selectively unfreeze the last N encoder+decoder layers
            unfreeze_last_n_whisper_layers(self.whisper, unfreeze_last_n)

    @property
    def generation_config(self):
        return self.whisper.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.whisper.generation_config = value

    @property
    def config(self):
        return self.whisper.config

    @config.setter
    def config(self, value):
        self.whisper.config = value

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    # -------------------------------------------------------------------------
    # Shared encoder-side computation
    # -------------------------------------------------------------------------
    def _pool(self, hidden: torch.Tensor, apply_proj: bool) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        if apply_proj and self.use_linear_proj:
            pooled = self.linear_proj(pooled)
        return pooled

    def _condition_and_route(self, hidden, logits_accbin, logits_accent):
        # Soft gate p(accented=1)
        p = torch.softmax(logits_accbin, dim=-1)[:, 1]
        p_t = p.view(-1, 1, 1).to(hidden.dtype)

        # Accent conditioning
        accent_probs = torch.softmax(logits_accent, dim=-1)
        e_accent = accent_probs @ self.accent_embed.weight
        e_neutral = self.neutral_embed.unsqueeze(0).expand_as(e_accent)
        e = (p.unsqueeze(1) * e_accent) + ((1.0 - p).unsqueeze(1) * e_neutral)

        e_t = e.unsqueeze(1).expand(hidden.size(0), hidden.size(1), hidden.size(2))
        hidden_cond = self.cond_proj(hidden + e_t)

        # Router/adapters; blend using p
        adapted_all, router_w = self.router(hidden_cond, self.adapters)
        encoder_hidden = p_t * adapted_all + (1.0 - p_t) * hidden
        return encoder_hidden, router_w

    def forward(
        self,
        input_features,
        decoder_input_ids=None,
        labels=None,
        accented_or_not_labels=None,
        gender_labels=None,
        accent_labels=None,
        **kwargs
    ):
        # 1) Encoder
        enc_out = self.whisper.model.encoder(input_features)
        hidden = enc_out.last_hidden_state
        pooled = self._pool(hidden, apply_proj=True)

        # 2) Heads
        logits_accbin = self.head_accented(pooled)
        logits_accent = self.head_accent(pooled)
        logits_gender = self.head_gender(self.grl(pooled))

        # 3) Classification losses
        loss_accbin = None
        loss_accent = None
        loss_gender = None

        if accented_or_not_labels is not None:
            loss_accbin = F.cross_entropy(logits_accbin, accented_or_not_labels)

        if gender_labels is not None:
            loss_gender = F.cross_entropy(logits_gender, gender_labels)

        if accent_labels is not None:
            loss_accent = F.cross_entropy(
                logits_accent,
                accent_labels,
                ignore_index=self.accent_ignore_index,
            )

        # 4-6) Accent conditioning + mixture of adapters
        encoder_hidden, router_w = self._condition_and_route(hidden, logits_accbin, logits_accent)

        # 7) Decoder
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.shift_tokens_right(
                labels,
                self.whisper.config.pad_token_id,
                self.whisper.config.decoder_start_token_id,
            )

        dec_out = self.whisper.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
        )
        logits_asr = self.whisper.proj_out(dec_out.last_hidden_state)

        # 8) ASR loss
        loss_asr = None
        if labels is not None:
            loss_asr = F.cross_entropy(
                logits_asr.reshape(-1, logits_asr.size(-1)),
                labels.reshape(-1),  # reshape: collator slicing makes labels non-contiguous on CPU
                ignore_index=-100,
            )

        # 9) Total loss
        parts = []
        if loss_asr is not None:
            parts.append(self.w_asr * loss_asr)
        if loss_accbin is not None:
            parts.append(self.w_accbin * loss_accbin)
        if loss_accent is not None:
            parts.append(self.w_accent * loss_accent)
        if loss_gender is not None:
            parts.append(self.w_gender * loss_gender)

        total_loss = sum(parts) if parts else None

        return {
            "loss": total_loss,
            "logits": logits_asr,
            "router_weights": router_w,
            "loss_asr": loss_asr,
            "loss_accented": loss_accbin,
            "loss_accent": loss_accent,
            "loss_gender": loss_gender,
            "logits_accented_or_not": logits_accbin,
            "logits_accent": logits_accent,
            "logits_gender": logits_gender,
        }

    @torch.no_grad()
    def generate(self, input_features, attention_mask=None, **generate_kwargs):
        generate_kwargs.pop("labels", None)
        generate_kwargs.pop("accented_or_not_labels", None)
        generate_kwargs.pop("gender_labels", None)
        generate_kwargs.pop("accent_labels", None)

        if attention_mask is None:
            attention_mask = torch.ones(
                (input_features.size(0), input_features.size(-1)),
                dtype=torch.long,
                device=input_features.device,
            )

        enc_out = self.whisper.model.encoder(input_features)
        hidden = enc_out.last_hidden_state
        pooled = self._pool(hidden, apply_proj=self.proj_in_generate)

        logits_accbin = self.head_accented(pooled)
        logits_accent = self.head_accent(pooled)

        encoder_hidden, _ = self._condition_and_route(hidden, logits_accbin, logits_accent)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        self.whisper.generation_config.forced_decoder_ids = None

        return self.whisper.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generate_kwargs
        )

    @torch.no_grad()
    def predict_clf(self, input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classification-only forward (NO decoder). Safe when labels are not available.
        Returns logits for accented_or_not (2), gender (2), accent (num_accents).
        """
        enc_out = self.whisper.model.encoder(input_features)
        hidden = enc_out.last_hidden_state
        pooled = self._pool(hidden, apply_proj=True)

        return {
            "logits_accbin": self.head_accented(pooled),
            "logits_gender": self.head_gender(self.grl(pooled)),
            "logits_accent": self.head_accent(pooled),
        }

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        """
        Saves:
          - Whisper weights/config (HF format) in `save_directory`
          - MoAA-specific weights in `wrapper.safetensors`
          - MoAA hyper-parameters in `wrapper_config.json`
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.whisper.save_pretrained(str(save_path), safe_serialization=safe_serialization)

        flat_sd = {}

        def add(prefix: str, sd: Dict[str, torch.Tensor]):
            for k, v in sd.items():
                flat_sd[f"{prefix}.{k}"] = v.contiguous()

        add("head_accented", self.head_accented.state_dict())
        add("head_accent", self.head_accent.state_dict())
        add("head_gender", self.head_gender.state_dict())
        add("accent_embed", self.accent_embed.state_dict())
        flat_sd["neutral_embed"] = self.neutral_embed.detach().contiguous()
        add("cond_proj", self.cond_proj.state_dict())
        add("adapters", self.adapters.state_dict())
        add("router", self.router.state_dict())
        if self.use_linear_proj:
            add("linear_proj", self.linear_proj.state_dict())

        extra_path = save_path / ("wrapper.safetensors" if safe_serialization else "wrapper.bin")
        if safe_serialization:
            st.save_file(flat_sd, str(extra_path))
        else:
            torch.save(flat_sd, str(extra_path))

        cfg = {
            "class": "WhisperAccentedConditionalAdapters",
            "num_adapters": len(self.adapters),
            "bottleneck_size": self.adapters["adapter_0"].down.out_features if len(self.adapters) else None,
            "num_accents": self.num_accents,
            "accent_ignore_index": self.accent_ignore_index,
            "grl_lambda": self.grl_lambda,
            "use_linear_proj": self.use_linear_proj,
        }
        (save_path / "wrapper_config.json").write_text(json.dumps(cfg, indent=2))


# =============================================================================
# Loading
# =============================================================================
def _prepare_whisper_for_wrapping(whisper: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
    """Untie proj_out from the decoder embeddings (needed when Whisper is wrapped)."""
    whisper.tie_weights = lambda: None
    whisper.proj_out.weight = nn.Parameter(whisper.proj_out.weight.clone().detach())
    return whisper


def build_moaa_model(
    model_name: str,
    num_adapters: int,
    bottleneck_size: int,
    num_accents: int,
    accent_ignore_index: int,
    grl_lambda: float,
    loss_weights,
    use_linear_proj: bool,
    unfreeze_last_n: int,
    proj_in_generate: bool = False,
    logger=None,
) -> WhisperAccentedConditionalAdapters:
    base_whisper = WhisperForConditionalGeneration.from_pretrained(model_name)
    if unfreeze_last_n > 0 and logger is not None:
        logger.info("Variant unfreezes last %d Whisper encoder/decoder layers.", unfreeze_last_n)
    model = WhisperAccentedConditionalAdapters(
        whisper_model=base_whisper,
        num_adapters=num_adapters,
        bottleneck_size=bottleneck_size,
        num_accents=num_accents,
        accent_ignore_index=accent_ignore_index,
        freeze_whisper=True,
        grl_lambda=grl_lambda,
        loss_weights=loss_weights,
        use_linear_proj=use_linear_proj,
        unfreeze_last_n=unfreeze_last_n,
        proj_in_generate=proj_in_generate,
    )
    _prepare_whisper_for_wrapping(model.whisper)
    return model


def load_moaa_checkpoint(
    ckpt_dir: str,
    base_model_name: str = "openai/whisper-small",
    proj_in_generate: bool = False,
    device: str = "cpu",
    logger=None,
) -> WhisperAccentedConditionalAdapters:
    """
    Load a MoAA model from either checkpoint format produced by `scripts/train.py`:

      (a) HF Trainer checkpoint dir (`checkpoint-<step>/model.safetensors`):
          full state dict of the wrapper (Whisper + all MoAA modules). This is the
          best-WER model selected on the dev set, and is the RECOMMENDED format.

      (b) `last_checkpoint/` dir written by `save_pretrained` (Whisper HF weights +
          `wrapper.safetensors` + `wrapper_config.json`).
          NOTE: checkpoints released from the original runs did not store `linear_proj`
          in `wrapper.safetensors`; ASR output is unaffected (generate() does not use it),
          but classifier predictions from such a checkpoint are not meaningful. Use (a).

    Architecture hyper-parameters are inferred from the weights.
    """
    ckpt = Path(ckpt_dir)
    trainer_weights = ckpt / "model.safetensors"
    wrapper_weights = ckpt / "wrapper.safetensors"

    if trainer_weights.exists() and not wrapper_weights.exists():
        full_sd = st.load_file(str(trainer_weights), device=device)
        if not any(k.startswith("whisper.") for k in full_sd):
            raise ValueError(f"{trainer_weights} does not look like a MoAA Trainer checkpoint.")
        prefix_whisper = "whisper."
        whisper_cfg = WhisperConfig.from_pretrained(base_model_name)
        base = WhisperForConditionalGeneration(whisper_cfg)
        base.generation_config = GenerationConfig.from_pretrained(base_model_name)
        sd = full_sd
    elif wrapper_weights.exists():
        base = WhisperForConditionalGeneration.from_pretrained(str(ckpt))
        sd = st.load_file(str(wrapper_weights), device=device)
        prefix_whisper = None
    else:
        raise FileNotFoundError(f"No model.safetensors or wrapper.safetensors found in {ckpt}")

    num_adapters = len({k.split(".")[1] for k in sd if k.startswith("adapters.")})
    bottleneck_size = sd["adapters.adapter_0.down.weight"].shape[0]
    num_accents = sd["head_accent.weight"].shape[0]
    use_linear_proj = "linear_proj.weight" in sd

    wrapper_cfg_path = ckpt / "wrapper_config.json"
    grl_lambda, accent_ignore_index = 1.0, -100
    if wrapper_cfg_path.exists():
        cfg = json.loads(wrapper_cfg_path.read_text())
        grl_lambda = cfg.get("grl_lambda", 1.0)
        accent_ignore_index = cfg.get("accent_ignore_index", -100)
        # Older wrapper files dropped linear_proj; honour the config flag if present
        use_linear_proj = cfg.get("use_linear_proj", use_linear_proj)

    model = WhisperAccentedConditionalAdapters(
        whisper_model=base,
        num_adapters=num_adapters,
        bottleneck_size=bottleneck_size,
        num_accents=num_accents,
        accent_ignore_index=accent_ignore_index,
        grl_lambda=grl_lambda,
        use_linear_proj=use_linear_proj,
        proj_in_generate=proj_in_generate,
    )
    _prepare_whisper_for_wrapping(model.whisper)

    if prefix_whisper is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading {trainer_weights}: {unexpected[:10]}")
        if missing:
            raise RuntimeError(f"Missing keys when loading {trainer_weights}: {missing[:10]}")
    else:
        def load_into(module, prefix):
            sub = {k.split(prefix + ".", 1)[1]: v for k, v in sd.items() if k.startswith(prefix + ".")}
            module.load_state_dict(sub)

        load_into(model.head_accented, "head_accented")
        load_into(model.head_accent, "head_accent")
        load_into(model.head_gender, "head_gender")
        load_into(model.accent_embed, "accent_embed")
        load_into(model.cond_proj, "cond_proj")
        load_into(model.adapters, "adapters")
        load_into(model.router, "router")
        if "neutral_embed" in sd:
            model.neutral_embed.data.copy_(sd["neutral_embed"])
        if use_linear_proj:
            if any(k.startswith("linear_proj.") for k in sd):
                load_into(model.linear_proj, "linear_proj")
            elif logger is not None:
                logger.warning(
                    "linear_proj weights not found in %s (legacy wrapper file). ASR decoding is "
                    "unaffected, but classifier predictions will be random. Prefer a checkpoint-<step>/ dir.",
                    wrapper_weights,
                )

    if logger is not None:
        logger.info(
            "Loaded MoAA from %s | adapters=%d bottleneck=%d accents=%d linear_proj=%s",
            ckpt, num_adapters, bottleneck_size, num_accents, use_linear_proj,
        )
    return model
