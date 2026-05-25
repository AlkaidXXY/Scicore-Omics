#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
export PYTHONPATH="/data2/xiaoxinyu/project/pretrain-gene/my_custom_model/llava_compat:/data1/xiaoxinyu/SOTAModel/cellwhisperer/modules/LLaVA:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

DATA_JSON="/data2/xiaoxinyu/project/pretrain-gene/cellwhisperer_dlpfc_sft/dlpfc_cellwhisperer_conversations.json"
IMAGE_DATA="/data2/xiaoxinyu/project/pretrain-gene/cellwhisperer_dlpfc_sft/dlpfc_cellwhisperer_features.npz"
BASE_MODEL="/data1/xiaoxinyu/SOTAModel/cellwhisperer/results/models/llava"
PYTHON="/home/xiaoxinyu/miniconda3/envs/llava/bin/python"
OUTPUT_PATH="/data2/xiaoxinyu/project/pretrain-gene/sft_output/cellwhisperer_dlpfc_full_desc_$(date +%m%d_%H%M)"
LOG_PATH="/data2/xiaoxinyu/project/pretrain-gene/logs/cellwhisperer_sft_full_desc_$(date +%m%d_%H%M).log"

mkdir -p "$(dirname "$LOG_PATH")" "$OUTPUT_PATH"

echo "Starting CellWhisperer LLaVA SFT" | tee -a "$LOG_PATH"
echo "Model: $BASE_MODEL" | tee -a "$LOG_PATH"
echo "Data: $DATA_JSON" | tee -a "$LOG_PATH"
echo "Features: $IMAGE_DATA" | tee -a "$LOG_PATH"
echo "Output: $OUTPUT_PATH" | tee -a "$LOG_PATH"

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  /data1/xiaoxinyu/SOTAModel/cellwhisperer/modules/LLaVA/llava/train/train.py \
  --data_path "$DATA_JSON" \
  --image_data "$IMAGE_DATA" \
  --output_dir "$OUTPUT_PATH" \
  --model_name_or_path "$BASE_MODEL" \
  --version mistral_instruct \
  --mm_projector_type mlp2x_8t_gelu \
  --mm_vision_select_layer -1 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --group_by_modality_length False \
  --bf16 True \
  --num_train_epochs 6 \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 117 \
  --save_total_limit 10 \
  --learning_rate 5e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --logging_steps 20 \
  --tf32 True \
  --model_max_length 1024 \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --dataloader_num_workers 4 \
  --report_to none \
  --lazy_preprocess True \
  --lora_enable True \
  --lora_r 8 \
  --lora_alpha 32 2>&1 | tee -a "$LOG_PATH"
