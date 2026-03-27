#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python playground/bench_efficiency.py \
    --model_path "lmms-lab/LLaVA-Video-7B-Qwen2" \
    --dataset_jsonl "videomme.jsonl" \
    --limit 100 \
    --shuffle \
    --num_frames 64 \
    --num_warmup 1 \
    --num_runs 3 \
    --max_new_tokens 16 \
    --baseline_output "logs/efficiency/baseline_llava_vid.jsonl" \
    --flashvid_output "logs/efficiency/flashvid_llava_vid.jsonl" \
    --summary_output_json "logs/efficiency/summary_llava_vid.json"
