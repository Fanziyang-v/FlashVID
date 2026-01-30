from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    repeat_kv,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from .utils import fastv_prune
from .configuration_flashvid import FlashVidConfig


def Qwen2Model_forward(
    self: Qwen2Model,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # Obtain FlashVID config
    if not hasattr(self, "flashvid_config"):
        raise ValueError("FlashVID configuration is not set in the model.")
    flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
    is_prefill = hidden_states.shape[1] > 1

    assert all(decoder_layer.attention_type == "full_attention" for decoder_layer in self.layers[: self.config.num_hidden_layers])
    causal_mask = causal_mask_mapping["full_attention"]
    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        # Only prunes visual tokens in prefilling stage.
        if is_prefill:
            if layer_idx == flashvid_config.pruning_layer - 1:
                kwargs["output_attentions"] = True
            elif layer_idx == flashvid_config.pruning_layer:
                kwargs["output_attentions"] = False
                attn: torch.Tensor = attn_weights
                (
                    hidden_states,
                    causal_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                    _,
                ) = fastv_prune(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    attentions=attn,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    flashvid_config=flashvid_config,
                )
        hidden_states, attn_weights = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


def Qwen2DecoderLayer_forward(
    self: Qwen2DecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[TransformersKwargs],
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    # ! Return attention weights for deep layer token pruning.
    return hidden_states, attn_weights


def Qwen2Attention_forward(
    self: Qwen2Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # main diff with Llama
        **kwargs,
    )

    if kwargs.get("output_attentions", False) and attn_weights is None:
        # Calculate attention weights manually if not provided
        last_query = query_states[:, :, -1:, :]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(last_query, key_states.transpose(2, 3)) * self.scaling
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
