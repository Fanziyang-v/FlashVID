#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Evaluation benchmarks.
TASKS=("videomme" "egoschema" "mvbench" "longvideobench_val_v" "mlvu_test")

# Pretrained model path.
PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"

# Model arguments.
MAX_NUM_FRAMES=32
# * Configurable pixel constraints
MIN_PIXELS=50716 # 64*28*28
MAX_PIXELS=200704 # 256*28*28
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,attn_implementation=$ATTN_IMPLEMENTATION"

MODEL_ARGS="enable_flashvid=False,$BASE_MODEL_ARGS"
for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    accelerate launch \
    --main_process_port 18888 \
    --num_processes 8 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args $MODEL_ARGS \
    --tasks $task \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "qwen2_5_vl" \
    --output_path ./logs/baseline
done
