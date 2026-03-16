#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Evaluation benchmarks.
TASKS=("videomme" "mlvu_test" "longvideobench_val_v" "egoschema")

# Pretrained model path.
PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"

# ! FlashVid arguments.
RETENTION_RATIOS=(0.100 0.200 0.250 0.333) # 10% 20% 25% 33.3%
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
MAX_NUM_FRAMES=(160 80 64 48) # 10x 5x 4x 3x
MIN_PIXELS=50716 # 64*28*28
MAX_PIXELS=200704 # 256*28*28
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,attn_implementation=$ATTN_IMPLEMENTATION,min_pixels=$MIN_PIXELS,max_pixels=$MAX_PIXELS"


for idx in "${!RETENTION_RATIOS[@]}"; do
    retention_ratio=${RETENTION_RATIOS[$idx]}
    max_frames_num=${MAX_NUM_FRAMES[$idx]}
    echo "Running with retention_ratio=${retention_ratio}, max_num_frames=${max_frames_num}"
    MODEL_ARGS="$BASE_MODEL_ARGS,$BASE_FLASHVID_ARGS,retention_ratio=${retention_ratio},max_num_frames=${max_frames_num}"
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
    echo "Finished running with retention_ratio=${retention_ratio}, max_frames_num=${max_frames_num}"
done
