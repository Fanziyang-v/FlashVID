from typing import Optional

from dataclasses import dataclass, field


@dataclass
class FlashVidConfig:
    # Average retention ratio.
    retention_ratio: float = field(default=0.25)

    # 1) Token Selection Method. Defaults to ADTS.
    alpha: float = field(default=0.7) # Ratio of ADTS tokens.
    token_selection_method: str = field(default="attn_div")

    # 2) Tree-based Spatio-Temporal Token Merging.
    temporal_threshold: float = field(default=0.8)

    # Dynamic Video Segmentation (DySeg).
    do_segment: bool = field(default=True)
    segment_threshold: float = field(default=0.9)
    min_segment_num: int = field(default=8)
    complementary_segment: bool = field(default=True)

    # Vision-Side Compression params.
    num_attn_div_tokens: Optional[int] = field(default=None)
    num_sttm_tokens: Optional[int] = field(default=None)

    # Inner-LLM Compression params.
    visual_token_start_index: Optional[int] = field(default=None)
    visual_token_length: Optional[int] = field(default=None)
    expansion: float = field(default=1.25)
    pruning_layer: int = field(default=20)
    llm_retention_ratio: float = field(default=0.3)
