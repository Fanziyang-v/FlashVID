from typing import Optional, Tuple

import torch
import torch.nn as nn
from llava.model.multimodal_encoder.siglip_encoder import (
    SigLipAttention,
    SigLipEncoderLayer,
    SigLipVisionTower,
)



@torch.no_grad()
def SigLipVisionTower_forward(self: SigLipVisionTower, images: torch.Tensor):
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"Unexpected data type of images: {type(images)}. Only support torch.Tensor Now.")
    image_forward_outs = self.vision_tower(
        images.to(device=self.device, dtype=self.dtype),
        output_attentions=True,
        output_hidden_states=True,
    )
    image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
    cls_attentions = image_forward_outs.attentions[-1].to(images.dtype)
    assert image_features.shape[-2] == 729

    return image_features, cls_attentions


def SigLipAttention_forward(
    self: SigLipAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    batch_size, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    k_v_seq_len = key_states.shape[-2]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

    if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
        raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

    if attention_mask is not None:
        if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
            raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights.mean(1).mean(1)
