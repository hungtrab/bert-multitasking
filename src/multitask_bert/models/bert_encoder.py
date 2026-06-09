"""Self-implemented BERT encoder used by the multitask model.

The first v2 scaffold used ``transformers.AutoModel`` directly. That is useful
for quick experiments, but it is not acceptable when the course requirement is
to implement BERT ourselves. This module keeps the same public contract while
replacing the encoder with a local implementation of BERT-base:

``token_ids, attention_mask -> EncoderOutput(cls, sequence)``

The model can still load ``bert-base-uncased`` config/weights from the
HuggingFace repository format. Only the parameter files are reused; the forward
pass is implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class EncoderOutput:
    cls: torch.Tensor          # (B, hidden)
    sequence: torch.Tensor     # (B, L, hidden)


@dataclass
class MiniBertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "MiniBertConfig":
        """Load only the BERT config metadata.

        Using ``AutoConfig`` here is fine: it gives dimensions/dropout values
        without outsourcing the model implementation.
        """
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(name_or_path)
        return cls(
            vocab_size=int(cfg.vocab_size),
            hidden_size=int(cfg.hidden_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            intermediate_size=int(cfg.intermediate_size),
            hidden_dropout_prob=float(cfg.hidden_dropout_prob),
            attention_probs_dropout_prob=float(cfg.attention_probs_dropout_prob),
            max_position_embeddings=int(cfg.max_position_embeddings),
            type_vocab_size=int(cfg.type_vocab_size),
            initializer_range=float(cfg.initializer_range),
            layer_norm_eps=float(cfg.layer_norm_eps),
            pad_token_id=int(cfg.pad_token_id),
        )


def build_tokenizer(name: str = "bert-base-uncased"):
    """Build the tokenizer compatible with the pretrained BERT vocabulary.

    The encoder is self-implemented; the tokenizer is still loaded from the
    pretrained vocabulary so token IDs match the checkpoint.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


class BertSelfAttention(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.transpose(1, 2)  # (B, heads, L, head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        key = self._split_heads(self.key(hidden_states))
        query = self._split_heads(self.query(hidden_states))
        value = self._split_heads(self.value(hidden_states))

        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, value)
        context = context.transpose(1, 2).contiguous()
        bsz, seq_len, _, _ = context.shape
        return context.view(bsz, seq_len, self.all_head_size)


class BertLayer(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.self_attention = BertSelfAttention(config)

        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    @staticmethod
    def _add_norm(
        residual: torch.Tensor,
        output: torch.Tensor,
        dense: nn.Linear,
        dropout: nn.Dropout,
        layer_norm: nn.LayerNorm,
    ) -> torch.Tensor:
        return layer_norm(residual + dropout(dense(output)))

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_out = self.self_attention(hidden_states, attention_mask)
        attention_out = self._add_norm(
            hidden_states,
            attention_out,
            self.attention_dense,
            self.attention_dropout,
            self.attention_layer_norm,
        )
        feedforward = F.gelu(self.interm_dense(attention_out))
        return self._add_norm(
            attention_out,
            feedforward,
            self.out_dense,
            self.out_dropout,
            self.out_layer_norm,
        )


class MiniBertModel(nn.Module):
    """Minimal BERT encoder matching the checkpoint key structure used in v1."""

    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.config = config

        self.word_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids, persistent=False)

        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def _extended_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        # Input mask: 1 for real tokens, 0 for padding.
        extended = attention_mask[:, None, None, :].to(dtype=torch.float32)
        return (1.0 - extended) * -10000.0

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        word_embeds = self.word_embedding(input_ids)
        return self.embed_from_word_embeddings(word_embeds)

    def embed_from_word_embeddings(self, word_embeds: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = word_embeds.shape
        pos_ids = self.position_ids[:, :seq_len].expand(bsz, seq_len)
        token_type_ids = torch.zeros(
            (bsz, seq_len),
            dtype=torch.long,
            device=word_embeds.device,
        )
        pos_embeds = self.pos_embedding(pos_ids)
        token_type_embeds = self.tk_type_embedding(token_type_ids)
        hidden = word_embeds + pos_embeds + token_type_embeds
        return self.embed_dropout(self.embed_layer_norm(hidden))

    def encode(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        extended_mask = self._extended_attention_mask(attention_mask).to(hidden_states.device)
        extended_mask = extended_mask.to(dtype=hidden_states.dtype)
        for layer in self.bert_layers:
            hidden_states = layer(hidden_states, extended_mask)
        return hidden_states

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EncoderOutput:
        sequence = self.encode(self.embed(input_ids), attention_mask)
        pooled = self.pooler_af(self.pooler_dense(sequence[:, 0]))
        return EncoderOutput(cls=pooled, sequence=sequence)

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> EncoderOutput:
        sequence = self.encode(self.embed_from_word_embeddings(embeddings), attention_mask)
        pooled = self.pooler_af(self.pooler_dense(sequence[:, 0]))
        return EncoderOutput(cls=pooled, sequence=sequence)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embedding(token_ids)


_HF_TO_LOCAL_REPLACEMENTS = (
    ("embeddings.word_embeddings", "word_embedding"),
    ("embeddings.position_embeddings", "pos_embedding"),
    ("embeddings.token_type_embeddings", "tk_type_embedding"),
    ("embeddings.LayerNorm", "embed_layer_norm"),
    ("encoder.layer", "bert_layers"),
    ("attention.self", "self_attention"),
    ("attention.output.dense", "attention_dense"),
    ("attention.output.LayerNorm", "attention_layer_norm"),
    ("intermediate.dense", "interm_dense"),
    ("output.dense", "out_dense"),
    ("output.LayerNorm", "out_layer_norm"),
    ("pooler.dense", "pooler_dense"),
)


def _map_hf_key(key: str) -> str | None:
    if key.startswith("bert."):
        key = key[len("bert."):]
    if key.startswith("cls."):
        return None
    if key.endswith(".gamma"):
        key = key.removesuffix(".gamma") + ".weight"
    elif key.endswith(".beta"):
        key = key.removesuffix(".beta") + ".bias"
    for old, new in _HF_TO_LOCAL_REPLACEMENTS:
        key = key.replace(old, new)
    return key


def _load_state_dict_from_hf(name_or_path: str) -> dict[str, torch.Tensor]:
    """Load raw checkpoint tensors without instantiating a HuggingFace model."""
    path = Path(name_or_path)
    if path.is_dir():
        safetensors_path = path / "model.safetensors"
        bin_path = path / "pytorch_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file

            return load_file(str(safetensors_path))
        if bin_path.exists():
            return torch.load(bin_path, map_location="cpu")

    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        weight_file = hf_hub_download(name_or_path, filename="model.safetensors")
        return load_file(weight_file)
    except Exception:
        from huggingface_hub import hf_hub_download

        weight_file = hf_hub_download(name_or_path, filename="pytorch_model.bin")
        return torch.load(weight_file, map_location="cpu")


def _load_pretrained_weights(model: MiniBertModel, name_or_path: str) -> None:
    raw_state = _load_state_dict_from_hf(name_or_path)
    local_state = model.state_dict()
    mapped_state: dict[str, torch.Tensor] = {}

    for key, value in raw_state.items():
        mapped = _map_hf_key(key)
        if mapped is None:
            continue
        if mapped in local_state and local_state[mapped].shape == value.shape:
            mapped_state[mapped] = value

    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    meaningful_missing = [k for k in missing if not k.endswith("position_ids")]
    if meaningful_missing:
        raise RuntimeError(
            "Could not load all BERT weights into the local implementation. "
            f"Missing keys: {meaningful_missing[:8]}"
        )
    if unexpected:
        raise RuntimeError(f"Unexpected BERT weight keys: {unexpected[:8]}")


class BertEncoder(nn.Module):
    """Public wrapper used by the rest of the multitask codebase."""

    def __init__(
        self,
        name: str = "bert-base-uncased",
        *,
        freeze: bool = False,
        load_pretrained: bool = True,
    ):
        super().__init__()
        config = MiniBertConfig.from_pretrained(name)
        self.bert = MiniBertModel(config)
        self.hidden_size = config.hidden_size

        if load_pretrained:
            _load_pretrained_weights(self.bert, name)
        self.set_frozen(freeze)

    def set_frozen(self, frozen: bool) -> None:
        for param in self.bert.parameters():
            param.requires_grad = not frozen

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> EncoderOutput:
        return self.bert(token_ids, attention_mask)

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> EncoderOutput:
        return self.bert.forward_from_embeddings(embeddings, attention_mask)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.bert.embed_tokens(token_ids)
