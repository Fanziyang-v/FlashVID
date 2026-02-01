from typing import Callable, Optional, Union, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    Qwen2_5_VLAttention,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModelOutputWithPast,
    repeat_kv,
)
from .configuration_flashvid import FlashVidConfig
from .utils import fastv_prune, flashvid_compression


def Qwen2_5_VLTextModel_forward(
    self: Qwen2_5_VLTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
    # where each dim indicates visual spatial positions for temporal/height/width grids.
    # There are two scenarios when FA2-like packed masking might be activated.
    # 1. User specifically passed packed `position_ids` and no attention mask.
    #    In this case we expect the useer to create correct position ids for all 3 grids
    #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
    # 2. User runs forward with no attention mask and no position ids. In this case, position ids
    #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
    #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
    #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
        text_position_ids = None

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
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

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    # Obtain FlashVid config
    if not hasattr(self, "flashvid_config"):
        raise ValueError("FlashVid configuration is not set in the model.")
    flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
    is_prefill = hidden_states.shape[1] > 1

    assert all(decoder_layer.attention_type == "full_attention" for decoder_layer in self.layers[: self.config.num_hidden_layers])
    _output_attentions = output_attentions
    causal_mask = causal_mask_mapping["full_attention"]
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # Only prunes visual tokens at prefilling stage.
        if is_prefill:
            if layer_idx == flashvid_config.pruning_layer - 1:
                output_attentions = True
            elif layer_idx == flashvid_config.pruning_layer:
                output_attentions = _output_attentions
                attn = layer_outputs[1]
                (
                    hidden_states,
                    causal_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                    keep_indices,
                ) = fastv_prune(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    attentions=attn,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    flashvid_config=flashvid_config,
                )
                # Don't forget to update text_position_ids (otherwise may occur CUDA error)
                text_position_ids = text_position_ids[..., keep_indices].contiguous()
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if _output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def Qwen2_5_VisionTransformerPretrainedModel_forward(
    self: Qwen2_5_VisionTransformerPretrainedModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    num_blocks = len(self.blocks)
    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        # * VisionZip needs `attn_weights` and `attn_key` metric for compression.
        return_logits = (num_blocks - 1) == layer_num
        hidden_states, attn_weights = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            return_logits=return_logits,
        )

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    # * Process attn_weights and metric.
    # attn_weights = F.avg_pool2d(attn_weights.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
    num_frames = grid_thw[0][0].item()
    # print(f"Attn weights Shape: {attn_weights.shape}")
    seq_len = attn_weights.shape[-1] // 4
    attn_weights = attn_weights.view(num_frames, seq_len, -1).mean(-1).view(-1)
    attn_weights = attn_weights[reverse_indices].view(num_frames, -1)

    return hidden_states, attn_weights


def Qwen2_5_VLVisionBlock_forward(
    self: Qwen2_5_VLVisionBlock,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_logits: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    residual = hidden_states
    hidden_states, attn_weights = self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        return_logits=return_logits,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.mlp(self.norm2(hidden_states))
    hidden_states = residual + hidden_states

    return hidden_states, attn_weights


def Qwen2_5_VLAttention_forward(
    self: Qwen2_5_VLAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
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
        sliding_window=self.sliding_window,
        position_ids=position_ids,  # pass positions for FA2
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if output_attentions and attn_weights is None:
        # * Calculate attention weights manually if not provided
        last_query = query_states[:, :, -1:, :]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        # key_states = key_states.transpose(1, 2)
        attn_weights = torch.matmul(last_query, key_states.transpose(2, 3)) / self.head_dim**0.5
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    return attn_output, attn_weights


@torch.no_grad()
def Qwen2_5_VLForConditionalGeneration_generate(
    self: Qwen2_5_VLForConditionalGeneration,
    **kwargs,
):
    flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
    # Obtain the visual token start index and length
    visual_token_start_index = torch.where(kwargs["input_ids"][0] == self.config.video_token_id)[0][0].item()
    visual_token_length = torch.where(kwargs["input_ids"][0] == self.config.video_token_id)[0].shape[0]
    # Update FlashVid Config.
    flashvid_config.visual_token_start_index = visual_token_start_index
    flashvid_config.visual_token_length = visual_token_length

    return self.generate_ori(**kwargs)


def Qwen2_5_VLModel_forward(
    self: Qwen2_5_VLModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, cls_attention = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        n_video_tokens = video_embeds.shape[0]
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    ### Applies FlashVid compression here.
    if position_ids.shape[-1] > 1:
        num_frames, num_visual_tokens = cls_attention.shape
        flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
        video_features = video_embeds.view(num_frames, num_visual_tokens, -1)
        compressed_video_tokens, keep_visual_global_indices = flashvid_compression(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
        )
        visual_start_index = torch.where(input_ids[0] == self.config.video_token_id)[0][0].item()
        visual_length = n_video_tokens
        visual_end_index = visual_start_index + visual_length
        # Update FlashVid config.
        flashvid_config.visual_token_start_index = visual_start_index
        flashvid_config.visual_token_length = compressed_video_tokens.shape[0]
        # Filter `position_ids`, `attention_mask`, `inputs_embeds`
        global_indices = torch.arange(input_ids.shape[-1]).to(input_ids)
        keep_visual_global_indices += visual_start_index
        keep_global_indices = (
            torch.cat(
                [
                    global_indices[:visual_start_index],
                    keep_visual_global_indices,
                    global_indices[visual_end_index:],
                ],
                dim=0,
            )
            .sort()
            .values
        )
        bsz, _, hidden_size = inputs_embeds.shape
        inputs_embeds.scatter_(
            dim=1,
            index=keep_visual_global_indices.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, hidden_size),
            src=compressed_video_tokens.view(-1, hidden_size).unsqueeze(0),
        )
        inputs_embeds = torch.gather(
            inputs_embeds,
            dim=1,
            index=keep_global_indices.view(1, -1, 1).expand(bsz, -1, hidden_size),
        )
        position_ids = position_ids[:, :, keep_global_indices]
        attention_mask = attention_mask[:, keep_global_indices]
        cache_position = cache_position[keep_global_indices]
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def Qwen2_5_VLVisionAttention_forward(
    self: Qwen2_5_VLVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    return_logits: bool = False,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    assert self.config._attn_implementation == "flash_attention_2"
    # Flash Attention 2: Use cu_seqlens for variable length attention
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen,
        max_length_k=max_seqlen,
        is_causal=False,
        **kwargs,
    )

    attn_weights = None
    if return_logits:
        # Calculate attention weights manually.
        num_frames = cu_seqlens.shape[0] - 1
        q, k = query_states.squeeze(0), key_states.squeeze(0)
        # reshape to (seq_length, num_heads, head_dim)
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        q = q.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = attn_weights.mean(1).mean(1)
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output, attn_weights


def Qwen2_5_VLModel_get_video_features(
    self: Qwen2_5_VLModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    """
    Encodes videos into continuous embeddings that can be forwarded to the language model.

    Args:
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
    """
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    video_embeds, cls_attention = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
    split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds, cls_attention
