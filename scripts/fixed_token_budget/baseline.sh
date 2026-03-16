
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# Evaluation benchmarks.
TASKS=("videomme" "mlvu_test" "longvideobench_val_v" "egoschema")

# Pretrained model path.
PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"

# Model arguments.
MAX_NUM_FRAMES=16
MIN_PIXELS=50176 # min_piexels=64*28*28
MAX_PIXELS=200704 # max_piexels=512*28*28
ATTN_IMPLEMENTATION=flash_attention_2
MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,min_pixels=$MIN_PIXELS,max_pixels=$MAX_PIXELS"

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
    --output_path ./logs/fixed_token_budget/baseline
done
