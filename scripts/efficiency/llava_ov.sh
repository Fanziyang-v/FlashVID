#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python playground/bench_efficiency.py \
    --model_path "lmms-lab/llava-onevision-qwen2-7b-ov" \
    --dataset_jsonl "videomme.jsonl" \
    --limit 100 \
    --shuffle \
    --num_frames 64 \
    --num_warmup 1 \
    --num_runs 3 \
    --max_new_tokens 16 \
    --baseline_output "logs/efficiency/baseline_llava_ov.jsonl" \
    --flashvid_output "logs/efficiency/flashvid_llava_ov.jsonl" \
    --summary_output_json "logs/efficiency/summary_llava_ov.json"
