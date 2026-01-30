#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Evaluation benchmarks.
TASKS=("videomme" "mvbench" "longvideobench_val_v" "egoschema")

# Pretrained model path.
PRETRAINED="lmms-lab/LLaVA-Video-7B-Qwen2"

# ! FlashVid arguments.
RETENTION_RATIOS=(0.10 0.15 0.20 0.25)
## Dyseg (fixed)
DO_SEGMENT=True
MIN_SEGMENT_NUM=8
COMPLEMENTARY_SEGMENT=True
## ADTS and TSTM (fixed)
TOKEN_SELECTION_METHOD=attn_div_v2
TEMPORAL_THRESHOLD=0.8
ALPHA=0.7
## Inner-LLM Pruning (fixed)
EXPANSION=1.25
PRUNING_LAYER=20
LLM_RETENTION_RATIO=0.3

BASE_FLASHVID_ARGS="enable_flashvid=True,expansion=$EXPANSION,do_segment=$DO_SEGMENT,min_segment_num=$MIN_SEGMENT_NUM,complementary_segment=$COMPLEMENTARY_SEGMENT,token_selection_method=$TOKEN_SELECTION_METHOD,alpha=$ALPHA,temporal_threshold=$TEMPORAL_THRESHOLD,pruning_layer=$PRUNING_LAYER,llm_retention_ratio=$LLM_RETENTION_RATIO"

# Model arguments.
MAX_FRAMES_NUM=64
CONV_TEMPLATE=qwen_1_5
FORCE_SAMPLE=True
ADD_TIME_INSTRUCTION=False
MM_SPATIAL_POOL_MODE=average # * Different from LLaVA-OneVision
MM_NEWLINE_POSITION=frame # Add newline token after each frame.
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,conv_template=$CONV_TEMPLATE,mm_spatial_pool_mode=$MM_SPATIAL_POOL_MODE,mm_newline_position=$MM_NEWLINE_POSITION,max_frames_num=$MAX_FRAMES_NUM,attn_implementation=$ATTN_IMPLEMENTATION,force_sample=$FORCE_SAMPLE,add_time_instruction=$ADD_TIME_INSTRUCTION"


for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,$BASE_FLASHVID_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        accelerate launch \
            --main_process_port 18888 \
            --num_processes 8 \
            -m lmms_eval \
            --model llava_vid \
            --model_args $MODEL_ARGS \
            --tasks $task \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "llava_vid" \
            --output_path ./logs/flashvid
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
