#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Evaluation benchmarks.
TASKS=("videomme" "mvbench" "longvideobench_val_v" "egoschema")

# Pretrained model path.
PRETRAINED="lmms-lab/llava-onevision-qwen2-7b-ov"

# Model arguments.
MAX_FRAMES_NUM=32
CONV_TEMPLATE=qwen_1_5
MM_SPATIAL_POOL_MODE=bilinear
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,conv_template=$CONV_TEMPLATE,mm_spatial_pool_mode=$MM_SPATIAL_POOL_MODE,max_frames_num=$MAX_FRAMES_NUM,attn_implementation=$ATTN_IMPLEMENTATION"

MODEL_ARGS="enable_flashvid=False,$BASE_MODEL_ARGS"
for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    accelerate launch \
        --main_process_port 18888 \
        --num_processes 8 \
        -m lmms_eval \
        --model llava_onevision \
        --model_args $MODEL_ARGS \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "llava_onevision" \
        --output_path ./logs/baseline
done
