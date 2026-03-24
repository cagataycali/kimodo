# SPDX-FileCopyrightText: Copyright (c) 2026 Cagatay Cali
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2/Qwen3 bidirectional model for LLM2Vec text encoding.
# Follows the same pattern as bidirectional_llama.py — disables causal masking
# so the model attends to all tokens (bidirectional) instead of left-to-right.
#
# Qwen3 models use Qwen2Config internally, so this handles both Qwen2 and Qwen3.

import torch
from peft import PeftModel
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model, Qwen2PreTrainedModel
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from transformers.utils import logging

from .utils import is_transformers_attn_greater_or_equal_4_43_1

logger = logging.get_logger(__name__)


class ModifiedQwen2Attention(Qwen2Attention):
    """Qwen2 attention with causal masking disabled for bidirectional encoding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2DecoderLayer(Qwen2DecoderLayer):
    """Qwen2 decoder layer using bidirectional (non-causal) attention."""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = ModifiedQwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen2BiModel(Qwen2Model):
    """Qwen2 model with bidirectional attention (no causal mask).

    Used as text encoder for Kimodo — encodes text prompts into embeddings
    that the diffusion backbone conditions on.
    """

    _no_split_modules = ["ModifiedQwen2DecoderLayer"]

    def __init__(self, config: Qwen2Config):
        if not is_transformers_attn_greater_or_equal_4_43_1():
            raise ValueError(
                "Qwen2BiModel requires transformers >= 4.43.1"
            )
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ModifiedQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        """Override causal mask to be fully bidirectional (zeros instead of triangular)."""
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # Key difference from causal: fill with zeros (attend to everything)
        causal_mask = torch.zeros(
            (sequence_length, target_length), dtype=dtype, device=device
        )
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class Qwen2BiForMNTP(Qwen2ForCausalLM):
    """Qwen2 bidirectional model for masked next token prediction (MNTP).

    Used for PEFT adapter loading when available.
    """

    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Qwen2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    def save_peft_model(self, path):
        self.model.save_pretrained(path)
