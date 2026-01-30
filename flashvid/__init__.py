from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
)

from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.siglip_encoder import (
    SigLipAttention,
    SigLipVisionTower,
)

from .configuration_flashvid import FlashVidConfig
from .llava_arch import (
    LlavaMetaForCausalLM_encode_images,
    LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal,
)
from .modeling_qwen2 import (
    Qwen2Attention_forward,
    Qwen2DecoderLayer_forward,
    Qwen2Model_forward,
)
from .siglip_encoder import SigLipAttention_forward, SigLipVisionTower_forward


def flashvid(
    model: nn.Module,
    retention_ratio: float = 0.25,
    # 1) DySeg params (DEFAULT)
    do_segment: bool = True,
    segment_threshold: float = 0.9,
    min_segment_num: int = 8,
    complementary_segment: bool = True,
    # 2) ADTS and TSTM params (KEY)
    token_selection_method: str = "attn_div_v2",
    alpha: float = 0.7,
    temporal_threshold: float = 0.8,
    # 3) Inner-LLM Compression params (DEFAULT)
    expansion: float = 1.25,
    pruning_layer: int = 20,
    llm_retention_ratio: float = 0.3,
) -> nn.Module:
    """Apply FlashVID to the model"""

    # Replace with custom methods.
    if type(model) is LlavaQwenForCausalLM:  ## For LLaVA-OneVision or LLaVA-Video
        LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_encode_images
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = (
            LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal
        )
        SigLipAttention.forward = SigLipAttention_forward
        SigLipVisionTower.forward = SigLipVisionTower_forward
        Qwen2Attention.forward = Qwen2Attention_forward
        Qwen2DecoderLayer.forward = Qwen2DecoderLayer_forward
        Qwen2Model.forward = Qwen2Model_forward
    else:
        raise NotImplementedError(f"FlashVID is not supported for {type(model)} yet.")

    # Create FlashVid config.
    flashvid_config = FlashVidConfig(
        retention_ratio=retention_ratio,
        do_segment=do_segment,
        segment_threshold=segment_threshold,
        min_segment_num=min_segment_num,
        complementary_segment=complementary_segment,
        alpha=alpha,
        token_selection_method=token_selection_method,
        temporal_threshold=temporal_threshold,
        expansion=expansion,
        pruning_layer=pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
    )

    # Store FlashVid Config in the model.
    setattr(model, "flashvid_config", flashvid_config)
    setattr(model.model, "flashvid_config", flashvid_config)

    return model
