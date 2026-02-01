from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VisionTransformerPretrainedModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
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

from .modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention_forward,
    Qwen2_5_VLModel_forward,
    Qwen2_5_VLTextModel_forward,
    Qwen2_5_VLModel_get_video_features,
    Qwen2_5_VLVisionAttention_forward,
    Qwen2_5_VLVisionBlock_forward,
    Qwen2_5_VisionTransformerPretrainedModel_forward,
    Qwen2_5_VLForConditionalGeneration_generate,
)

from .modeling_qwen3_vl import (
    Qwen3VLVisionAttention_forward,
    Qwen3VLVisionBlock_forward,
    Qwen3VLVisionModel_forward,
    Qwen3VLModel_forward,
    Qwen3VLTextAttention_forward,
    Qwen3VLTextDecoderLayer_forward,
    Qwen3VLTextModel_forward,
    Qwen3VLModel_get_image_features,
)

from .siglip_encoder import SigLipAttention_forward, SigLipVisionTower_forward


def flashvid(
    model: nn.Module,
    retention_ratio: float = 0.25,
    # 1) DySeg params (FIXED)
    do_segment: bool = True,
    segment_threshold: float = 0.9,
    min_segment_num: int = 8,
    complementary_segment: bool = True,
    # 2) ADTS and TSTM params
    token_selection_method: str = "attn_div_v2",
    alpha: float = 0.7,
    temporal_threshold: float = 0.8,
    # 3) Inner-LLM Compression params
    expansion: float = 1.25,
    pruning_layer: int = 20,
    llm_retention_ratio: float = 0.3,
) -> nn.Module:
    """Apply FlashVID to the model.

    Args:
        model (nn.Module): The model to apply FlashVID to.
        retention_ratio (float, optional): The retention ratio. Defaults to 0.25.
        do_segment (bool, optional): Whether to perform dynamic video segmentation. Defaults to True.
        segment_threshold (float, optional): The threshold for dynamic video segmentation. Defaults to 0.9.
        min_segment_num (int, optional): The minimum number of segments. Defaults to 8.
        complementary_segment (bool, optional): Whether to perform complementary segmentation. Defaults to True.
        token_selection_method (str, optional): The method for token selection. Defaults to "attn_div_v2".
        alpha (float, optional): The alpha for token selection. Defaults to 0.7.
        temporal_threshold (float, optional): The temporal threshold for token selection. Defaults to 0.8.
        expansion (float, optional): The expansion ratio for inner-LLM compression. Defaults to 1.25.
        pruning_layer (int, optional): The layer to prune. Defaults to 20.
        llm_retention_ratio (float, optional): The retention ratio for inner-LLM compression. Defaults to 0.3.

    Raises:
        NotImplementedError: If the model is not supported.

    Returns:
        nn.Module: The model with FlashVID applied.
    """

    # Replace with custom methods.
    if type(model) is LlavaQwenForCausalLM:  ## For LLaVA-OneVision or LLaVA-Video
        LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_encode_images
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal
        SigLipAttention.forward = SigLipAttention_forward
        SigLipVisionTower.forward = SigLipVisionTower_forward
        Qwen2Attention.forward = Qwen2Attention_forward
        Qwen2DecoderLayer.forward = Qwen2DecoderLayer_forward
        Qwen2Model.forward = Qwen2Model_forward
    elif type(model) is Qwen2_5_VLForConditionalGeneration:  ## For Qwen2.5-VL
        Qwen2_5_VLAttention.forward = Qwen2_5_VLAttention_forward
        Qwen2_5_VLModel.get_video_features = Qwen2_5_VLModel_get_video_features
        Qwen2_5_VLTextModel.forward = Qwen2_5_VLTextModel_forward
        Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward
        Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_forward
        Qwen2_5_VLVisionAttention.forward = Qwen2_5_VLVisionAttention_forward
        Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_forward
        Qwen2_5_VLForConditionalGeneration.generate_ori = Qwen2_5_VLForConditionalGeneration.generate
        Qwen2_5_VLForConditionalGeneration.generate = Qwen2_5_VLForConditionalGeneration_generate
    elif type(model) is Qwen3VLForConditionalGeneration:  ## For Qwen3-VL
        Qwen3VLVisionAttention.forward = Qwen3VLVisionAttention_forward
        Qwen3VLVisionBlock.forward = Qwen3VLVisionBlock_forward
        Qwen3VLVisionModel.forward = Qwen3VLVisionModel_forward
        Qwen3VLModel.forward = Qwen3VLModel_forward
        Qwen3VLTextAttention.forward = Qwen3VLTextAttention_forward
        Qwen3VLTextDecoderLayer.forward = Qwen3VLTextDecoderLayer_forward
        Qwen3VLTextModel.forward = Qwen3VLTextModel_forward
        Qwen3VLModel.get_image_features = Qwen3VLModel_get_image_features
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
    if type(model) in (Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration):
        setattr(model.model.language_model, "flashvid_config", flashvid_config)

    return model
