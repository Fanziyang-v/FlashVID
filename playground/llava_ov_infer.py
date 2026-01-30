import argparse

from dataclasses import dataclass, field
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle

from transformers.hf_argparser import HfArgumentParser

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")


@dataclass
class InferenceArguments:
    """Arguments for the LLaVA OneVision model."""

    question: str = field(default="What is happening in this video?")
    video_path: str = field(default="path/to/video.mp4")
    model_path: str = field(default="lmms-lab/llava-onevision-qwen2-7b-ov")
    num_frames: int = field(default=32)
    enable_flashvid: bool = field(default=False)


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(
        0, total_frame_num - 1, max_frames_num, dtype=int
    )
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


def get_model(model_path: str):
    """Load the model from the specified path."""
    model_name = get_model_name_from_path(model_path)
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
    }
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        **llava_model_args,
    )
    model.eval()
    return tokenizer, model, image_processor, max_length


def parse_args():
    """Parse command line arguments."""
    parser = HfArgumentParser((InferenceArguments))
    (arguments,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    return arguments


def main(args: InferenceArguments):
    # Load pretrained model.
    tokenizer, model, image_processor, max_length = get_model(args.model_path)

    if args.enable_flashvid:
        # ! Apply FlashVID here.
        from flashvid import flashvid

        model = flashvid(
            model,
            retention_ratio=0.10,
            do_segment=True,
            segment_threshold=0.9,
            min_segment_num=8,
            complementary_segment=True,
            alpha=0.70,
            token_selection_method="attn_div_v2",  # Use ADTSv2
            temporal_threshold=0.8,
            expansion=1.25,
            pruning_layer=20,
            llm_retention_ratio=0.3,
        )

    # load video and process frames
    video_frames = load_video(args.video_path, args.num_frames)
    print(f"Loaded {len(video_frames)} frames from the video.")
    image_tensors = []
    frames = (
        image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )
    image_tensors.append(frames)

    # Prepare inputs
    conv_template = "qwen_1_5"
    question = f"{DEFAULT_IMAGE_TOKEN}\n{args.question}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    print(f"Prompt question: {prompt_question}")

    input_ids = (
        tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to("cuda")
    )
    image_sizes = [frame.size for frame in video_frames]

    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=4096,
            modalities=["video"],
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(text_outputs[0])


if __name__ == "__main__":
    args = parse_args()
    main(args)
