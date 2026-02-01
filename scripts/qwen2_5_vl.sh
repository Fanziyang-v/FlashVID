#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Evaluation benchmarks.
TASKS=("videomme" "egoschema" "mvbench" "longvideobench_val_v" "mlvu_test")

# Pretrained model path.
PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"

# ! FlashVid arguments.
RETENTION_RATIOS=(0.10 0.15 0.20 0.25)
## Dyseg (fixed)
DO_SEGMENT=True
MIN_SEGMENT_NUM=4
COMPLEMENTARY_SEGMENT=True
## ADTS and TSTM (fixed)
TOKEN_SELECTION_METHOD=attn_div # * Use ADTSv1 for Qwen2.5-VL
ALPHA=0.70
TEMPORAL_THRESHOLD=0.8
## Inner-LLM Pruning (fixed)
EXPANSION=1.25
PRUNING_LAYER=20
LLM_RETENTION_RATIO=0.3

BASE_FLASHVID_ARGS="enable_flashvid=True,expansion=$EXPANSION,do_segment=$DO_SEGMENT,min_segment_num=$MIN_SEGMENT_NUM,complementary_segment=$COMPLEMENTARY_SEGMENT,token_selection_method=$TOKEN_SELECTION_METHOD,alpha=$ALPHA,temporal_threshold=$TEMPORAL_THRESHOLD,pruning_layer=$PRUNING_LAYER,llm_retention_ratio=$LLM_RETENTION_RATIO"

# Model arguments.
MAX_NUM_FRAMES=32
# * Configurable pixel constraints.
# MIN_PIXELS=50716 # 64*28*28
# MAX_PIXELS=200704 # 256*28*28
ATTN_IMPLEMENTATION=flash_attention_2
# BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,attn_implementation=$ATTN_IMPLEMENTATION"
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"


for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,$BASE_FLASHVID_ARGS,retention_ratio=${retention_ratio}"
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
        --output_path ./logs/flashvid
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
