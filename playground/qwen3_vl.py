from dataclasses import dataclass, field
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.hf_argparser import HfArgumentParser
from qwen_vl_utils import process_vision_info


@dataclass
class InferenceArguments:
    """Arguments for the Qwen2.5-VL model."""

    question: str = field(default="What is happening in this video?")
    video_path: str = field(default="path/to/video.mp4")
    model_path: str = field(default="Qwen/Qwen3-VL-4B-Instruct")
    num_frames: int = field(default=32)
    enable_flashvid: bool = field(default=False)
    max_new_tokens: int = field(default=2048)


def load_model(model_path: str):
    """Load the Qwen2.5-VL model from the specified path."""
    # You can set the maximum tokens for a video through the environment variable VIDEO_MAX_PIXELS
    # based on the maximum tokens that the model can accept.
    # export VIDEO_MAX_PIXELS = 32000 * 28 * 28 * 0.9
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


@torch.no_grad()
def inference(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    video_path: str,
    prompt: str,
    max_new_tokens: int = 2048,
    max_pixels: int = 256 * 28 * 28,
    min_pixels: int = 64 * 28 * 28,
    num_frames: int = 32,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "video": video_path,
                    "max_pixels": max_pixels,
                    "min_pixels": min_pixels,
                    "nframes": num_frames,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("video input:", videos[0].shape)
    num_frames, _, resized_height, resized_width = videos[0].shape
    print(
        "num of video tokens:",
        int(num_frames / 2 * resized_height / 28 * resized_width / 28),
    )
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return generated_text


def parse_args():
    parser = HfArgumentParser((InferenceArguments))
    (arguments,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    return arguments


def main(args: InferenceArguments):
    model, processor = load_model(args.model_path)

    if args.enable_flashvid:
        # ! Apply FlashVID
        from flashvid import flashvid

        model = flashvid(
            model,
            retention_ratio=0.10,
            do_segment=True,
            segment_threshold=0.9,
            min_segment_num=4,
            complementary_segment=True,
            alpha=0.70,
            token_selection_method="attn_div",  # Use ADTSv1
            temporal_threshold=0.8,
            pruning_layer=24,
            llm_retention_ratio=0.4,
        )

    model.eval()  # Set the model to evaluation mode

    # Run inference
    generated_text = inference(model, processor, args.video_path, args.question, args.max_new_tokens)
    print(f"Generated Answer: {generated_text}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
