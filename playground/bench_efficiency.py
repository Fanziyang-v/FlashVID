import copy
import json
import os
import random
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers.hf_argparser import HfArgumentParser

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")

LOG_KEYS = ("prefilling_ms", "ttft_ms")
SEPARATOR = "=" * 64


@dataclass
class BenchmarkArgs:
    model_path: str = field(default="lmms-lab/llava-onevision-qwen2-7b-ov")
    dataset_jsonl: str = field(default="videomme.jsonl")
    limit: int | None = field(default=None)
    shuffle: bool = field(default=False)
    num_frames: int = field(default=64)
    num_warmup: int = field(default=1)
    num_runs: int = field(default=3)
    max_new_tokens: int = field(default=16)
    baseline_output: str = field(default="logs/efficiency/baseline.jsonl")
    flashvid_output: str = field(default="logs/efficiency/flashvid.jsonl")
    summary_output_json: str = field(default="logs/efficiency/summary.json")


def load_video(video_path: str, max_frames_num: int) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    return vr.get_batch(frame_idx.tolist()).asnumpy()


def get_model(model_path: str):
    model_name = get_model_name_from_path(model_path)
    overwrite_config = (
        {
            "mm_spatial_pool_mode": "average",
            "mm_newline_position": "frame",
        }
        if model_path == "lmms-lab/LLaVA-Video-7B-Qwen2"
        else {}
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map="auto",
        attn_implementation="flash_attention_2",
        overwrite_config=overwrite_config,
        multimodal=True,
    )
    model.eval()
    return tokenizer, model, image_processor


def prepare_inputs(tokenizer, image_processor, video_frames, prompt_text: str):
    frames = (
        image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to("cuda")
    )
    attention_mask = torch.ones_like(input_ids)
    image_sizes = [frame.size for frame in video_frames]
    return input_ids, attention_mask, [frames], image_sizes


def load_dataset_samples(
    dataset_jsonl: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with Path(dataset_jsonl).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def resolve_videomme_video_path(video_id: str) -> str:
    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/")))
    base_dir = hf_home / "videomme" / "data"
    for suffix in (".mp4", ".MP4", ".mkv"):
        candidate = base_dir / f"{video_id}{suffix}"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"missing video for videoID={video_id} under {base_dir}")


def extract_choice_letter(text: str) -> str:
    match = re.search(r"[ABCD]", text.upper())
    return match.group(0) if match else ""


def decode_generated_output(
    tokenizer,
    output_ids: torch.Tensor,
    prompt_length: int,
) -> str:
    generated_ids = (
        output_ids[:, prompt_length:]
        if output_ids.shape[1] > prompt_length
        else output_ids
    )
    if generated_ids.shape[1] == 0:
        return ""

    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()
    pred_answer = extract_choice_letter(generated_text)
    if pred_answer:
        return pred_answer

    first_token_id = int(generated_ids[0, 0].item())
    first_token = tokenizer.decode([first_token_id], skip_special_tokens=True)
    if first_token:
        return extract_choice_letter(first_token)
    return extract_choice_letter(tokenizer.convert_ids_to_tokens(first_token_id))


def read_jsonl_records(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None}
    array = np.array(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
    }


def summarize_phase_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("error")]
    correctness = [
        float(record["correct"])
        for record in valid
        if record.get("correct") is not None
    ]
    return {
        "num_samples": len(records),
        "num_valid": len(valid),
        "num_errors": len(records) - len(valid),
        "accuracy": float(np.mean(correctness)) if correctness else None,
        "latency_ms": {
            key: _stats(
                [float(record[key]) for record in valid if record.get(key) is not None]
            )
            for key in LOG_KEYS
        },
    }


def summarize_speedups(
    baseline_records: list[dict[str, Any]],
    flashvid_records: list[dict[str, Any]],
) -> dict[str, Any]:
    flashvid_by_qid = {
        record["question_id"]: record
        for record in flashvid_records
        if record.get("question_id")
    }
    matched_pairs = [
        (baseline_record, flashvid_by_qid[baseline_record["question_id"]])
        for baseline_record in baseline_records
        if baseline_record.get("question_id") in flashvid_by_qid
        and not baseline_record.get("error")
        and not flashvid_by_qid[baseline_record["question_id"]].get("error")
    ]

    speedups: dict[str, Any] = {}
    for key in LOG_KEYS:
        ratios = [
            float(baseline_record[key]) / float(flashvid_record[key])
            for baseline_record, flashvid_record in matched_pairs
            if baseline_record.get(key) not in (None, 0)
            and flashvid_record.get(key) not in (None, 0)
        ]
        baseline_values = [
            float(baseline_record[key])
            for baseline_record, _ in matched_pairs
            if baseline_record.get(key) is not None
        ]
        flashvid_values = [
            float(flashvid_record[key])
            for _, flashvid_record in matched_pairs
            if flashvid_record.get(key) is not None
        ]
        overall_ratio = None
        if (
            baseline_values
            and flashvid_values
            and np.mean(baseline_values) > 0
            and np.mean(flashvid_values) > 0
        ):
            overall_ratio = float(np.mean(baseline_values) / np.mean(flashvid_values))
        speedups[key] = {
            "overall_ratio": overall_ratio,
            "per_sample": _stats(ratios),
        }

    return {"matched_samples": len(matched_pairs), "speedups": speedups}


def build_summary_from_logs(
    baseline_output: str,
    flashvid_output: str,
) -> dict[str, Any]:
    baseline_records = read_jsonl_records(baseline_output)
    flashvid_records = read_jsonl_records(flashvid_output)
    return {
        "baseline": summarize_phase_records(baseline_records),
        "flashvid": summarize_phase_records(flashvid_records),
        "comparison": summarize_speedups(baseline_records, flashvid_records),
    }


class Timer:
    def __init__(self):
        self._events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._originals: dict[str, Any] = {}
        self._finished_events: set[str] = set()

    def _create_events(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self._events[name] = (start, end)
        return start, end

    def install(self, model, *, time_compression: bool) -> None:
        """Install hooks that record only the first relevant forward pass."""

        vision_tower = model.get_model().get_vision_tower()
        mm_projector = model.get_model().mm_projector
        llm_backbone = model.model
        timer = self

        def _vision_pre(module, inp):
            if "vision" in timer._events:
                return
            start, _ = timer._create_events("vision")
            start.record()

        def _vision_post(module, inp, out):
            if "vision" not in timer._events or "vision" in timer._finished_events:
                return
            _, end = timer._events["vision"]
            end.record()
            timer._finished_events.add("vision")

        def _llm_pre(module, inp):
            if "llm_forward" in timer._events:
                return
            start, _ = timer._create_events("llm_forward")
            start.record()

        def _llm_post(module, inp, out):
            if (
                "llm_forward" not in timer._events
                or "llm_forward" in timer._finished_events
            ):
                return
            _, end = timer._events["llm_forward"]
            end.record()
            timer._finished_events.add("llm_forward")

        self._hooks.append(vision_tower.register_forward_pre_hook(_vision_pre))
        self._hooks.append(mm_projector.register_forward_hook(_vision_post))
        self._hooks.append(llm_backbone.register_forward_pre_hook(_llm_pre))
        self._hooks.append(llm_backbone.register_forward_hook(_llm_post))

        if time_compression:
            import flashvid.llava_arch as flashvid_arch

            original = flashvid_arch.flashvid_compression
            self._originals["compression"] = (flashvid_arch, original)

            def _timed_compression(*args, **kwargs):
                if "compression" in timer._events:
                    return original(*args, **kwargs)
                start, end = timer._create_events("compression")
                start.record()
                result = original(*args, **kwargs)
                end.record()
                return result

            flashvid_arch.flashvid_compression = _timed_compression

    def cleanup(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        if "compression" in self._originals:
            module, original = self._originals["compression"]
            module.flashvid_compression = original

        self._originals.clear()
        self._events.clear()
        self._finished_events.clear()

    def get_times_ms(self) -> dict[str, float]:
        torch.cuda.synchronize()
        return {
            name: start.elapsed_time(end) for name, (start, end) in self._events.items()
        }


@torch.inference_mode()
def run_benchmark(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    image_tensors,
    image_sizes,
    num_warmup: int,
    num_runs: int,
    max_new_tokens: int,
    *,
    use_flashvid: bool,
) -> dict[str, Any]:
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "modalities": ["video"],
    }

    for _ in range(num_warmup):
        model.generate(
            input_ids.clone(),
            attention_mask=attention_mask.clone(),
            images=[tensor.clone() for tensor in image_tensors],
            image_sizes=image_sizes,
            **gen_kwargs,
        )
        torch.cuda.synchronize()

    vision_times: list[float] = []
    compression_times: list[float] = []
    llm_times: list[float] = []
    pred_answer = ""

    for run_idx in range(num_runs):
        timer = Timer()
        timer.install(model, time_compression=use_flashvid)
        output_ids = model.generate(
            input_ids.clone(),
            attention_mask=attention_mask.clone(),
            images=[tensor.clone() for tensor in image_tensors],
            image_sizes=image_sizes,
            **gen_kwargs,
        )
        if run_idx == 0:
            pred_answer = decode_generated_output(
                tokenizer,
                output_ids,
                input_ids.shape[1],
            )

        times = timer.get_times_ms()
        timer.cleanup()
        vision_times.append(times.get("vision", 0.0))
        compression_times.append(times.get("compression", 0.0))
        llm_times.append(times.get("llm_forward", 0.0))
        torch.cuda.empty_cache()

    vision = np.array(vision_times)
    compression = np.array(compression_times)
    llm = np.array(llm_times)
    return {
        "pred_answer": pred_answer,
        "prefilling_ms": float(np.mean(compression + llm)),
        "ttft_ms": float(np.mean(vision + compression + llm)),
    }


def benchmark_single_sample(
    sample: dict[str, Any],
    tokenizer,
    model,
    image_processor,
    args: BenchmarkArgs,
    *,
    use_flashvid: bool,
) -> dict[str, Any]:
    record = {
        "question_id": sample.get("question_id"),
        "videoID": sample.get("videoID"),
        "answer": sample.get("answer"),
        "pred_answer": "",
        "correct": None,
        "prefilling_ms": None,
        "ttft_ms": None,
        "error": None,
    }

    try:
        video_path = resolve_videomme_video_path(sample["videoID"])
        video_frames = load_video(video_path, args.num_frames)
        input_ids, attention_mask, image_tensors, image_sizes = prepare_inputs(
            tokenizer,
            image_processor,
            video_frames,
            sample["input"],
        )
        result = run_benchmark(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            image_tensors,
            image_sizes,
            args.num_warmup,
            args.num_runs,
            args.max_new_tokens,
            use_flashvid=use_flashvid,
        )
        record["pred_answer"] = result["pred_answer"]
        record["correct"] = result["pred_answer"] == sample.get("answer")
        record["prefilling_ms"] = result["prefilling_ms"]
        record["ttft_ms"] = result["ttft_ms"]
    except Exception as exc:  # pragma: no cover - runtime failure path
        record["error"] = str(exc)

    return record


def run_dataset_phase(
    samples: list[dict[str, Any]],
    tokenizer,
    model,
    image_processor,
    args: BenchmarkArgs,
    *,
    use_flashvid: bool,
    output: str,
) -> None:
    phase_name = "FlashVID" if use_flashvid else "Baseline"
    total = len(samples)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for idx, sample in enumerate(samples, 1):
            record = benchmark_single_sample(
                sample,
                tokenizer,
                model,
                image_processor,
                args,
                use_flashvid=use_flashvid,
            )
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()

            if record["error"]:
                print(
                    f"[{phase_name}] {idx}/{total} "
                    f"{record['question_id']} error: {record['error']}"
                )
                continue
            print(
                f"[{phase_name}] {idx}/{total} {record['question_id']} "
                f"pred={record['pred_answer'] or '-'} gt={record['answer']} "
                f"correct={record['correct']} ttft={record['ttft_ms']:.2f}ms"
            )


def _print_header(args: BenchmarkArgs) -> None:
    print(SEPARATOR)
    print("  FlashVID Video-MME Benchmark")
    print(SEPARATOR)
    print(f"  Model      : {args.model_path}")
    print(f"  Dataset    : {args.dataset_jsonl}")
    print(f"  Limit      : {args.limit}")
    print(f"  Shuffle    : {args.shuffle}")
    print(f"  Frames     : {args.num_frames}")
    print(f"  Warmup     : {args.num_warmup}")
    print(f"  Timed runs : {args.num_runs}")
    print(f"  Max new tok: {args.max_new_tokens}")
    print(SEPARATOR)
    print()


def _print_dataset_summary(summary: dict[str, Any]) -> None:
    print(SEPARATOR)
    print("  Dataset Summary")
    print(SEPARATOR)
    for phase_name in ("baseline", "flashvid"):
        phase = summary[phase_name]
        accuracy = phase["accuracy"]
        accuracy_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
        print(f"  [{phase_name.capitalize()}]")
        print(f"    Samples   : {phase['num_samples']}")
        print(f"    Valid     : {phase['num_valid']}")
        print(f"    Errors    : {phase['num_errors']}")
        print(f"    Accuracy  : {accuracy_text}")
        for key in LOG_KEYS:
            stats = phase["latency_ms"][key]
            if stats["mean"] is None or stats["median"] is None:
                print(f"    {key:<14}: N/A")
                continue
            print(
                f"    {key:<14}: mean {stats['mean']:8.2f} ms"
                f" | median {stats['median']:8.2f} ms"
            )
        print()

    print("  [Speedup]")
    print(f"    Matched   : {summary['comparison']['matched_samples']}")
    for key in LOG_KEYS:
        stats = summary["comparison"]["speedups"][key]
        overall = stats["overall_ratio"]
        median = stats["per_sample"]["median"]
        if overall is None or median is None:
            print(f"    {key:<14}: N/A")
            continue
        print(
            f"    {key:<14}: overall {overall:.2f}x | per-sample median {median:.2f}x"
        )
    print(SEPARATOR)


def apply_flashvid(model):
    from flashvid import flashvid

    return flashvid(
        model,
        retention_ratio=0.10,
        do_segment=True,
        segment_threshold=0.9,
        min_segment_num=8,
        complementary_segment=True,
        alpha=0.70,
        token_selection_method="attn_div_v2",
        temporal_threshold=0.8,
        expansion=1.25,
        pruning_layer=20,
        llm_retention_ratio=0.3,
    )


def run_dataset_benchmark(args: BenchmarkArgs) -> None:
    samples = load_dataset_samples(args.dataset_jsonl, args.limit)
    if not samples:
        raise ValueError(f"no samples found in {args.dataset_jsonl}")
    if args.shuffle:
        random.shuffle(samples)

    print(f"Loaded {len(samples)} samples from {args.dataset_jsonl}.\n")
    tokenizer, model, image_processor = get_model(args.model_path)

    print("Benchmarking Baseline ...")
    run_dataset_phase(
        samples,
        tokenizer,
        model,
        image_processor,
        args,
        use_flashvid=False,
        output=args.baseline_output,
    )

    print("\nBenchmarking FlashVID ...")
    flashvid_model = apply_flashvid(model)
    run_dataset_phase(
        samples,
        tokenizer,
        flashvid_model,
        image_processor,
        args,
        use_flashvid=True,
        output=args.flashvid_output,
    )

    summary = build_summary_from_logs(
        args.baseline_output,
        args.flashvid_output,
    )
    summary_path = Path(args.summary_output_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_dataset_summary(summary)


def main() -> None:
    """Parse args and run the dataset benchmark."""

    parser = HfArgumentParser(BenchmarkArgs)
    (args,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    _print_header(args)
    run_dataset_benchmark(args)


if __name__ == "__main__":
    main()
