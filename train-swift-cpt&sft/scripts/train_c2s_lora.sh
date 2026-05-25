#!/bin/bash
set +u
source /home/xiaoxinyu/.bashrc
conda activate /home/xiaoxinyu/miniconda3/envs/pt

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

BASE_MODEL="/data1/xiaoxinyu/SOTAModel/C2S"
TRAIN_DATA="/data1/xiaoxinyu/project/data/format_dataset/DLPFC_tri_QA_balanced_train_v5.jsonl"
OUTPUT_PATH="/data2/xiaoxinyu/project/pretrain-gene/sft_output/c2s_dlpfc_full_desc_$(date +%m%d_%H%M)"
LOG_PATH="/data2/xiaoxinyu/project/pretrain-gene/logs/c2s_sft_full_desc_$(date +%m%d_%H%M).log"
CACHE_PATH="/data2/xiaoxinyu/project/pretrain-gene/cache/DLPFC_tri_QA_balanced_train_v5.c2s_full_desc.jsonl"

mkdir -p "$(dirname "$LOG_PATH")"

echo "Starting C2S LoRA SFT" | tee -a "$LOG_PATH"
echo "Model: $BASE_MODEL" | tee -a "$LOG_PATH"
echo "Dataset: $TRAIN_DATA" | tee -a "$LOG_PATH"
echo "Output: $OUTPUT_PATH" | tee -a "$LOG_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_PATH"

/home/xiaoxinyu/miniconda3/envs/pt/bin/python \
    /data2/xiaoxinyu/project/pretrain-gene/my_custom_model/c2s_lora_sft.py \
    --model "$BASE_MODEL" \
    --dataset "$TRAIN_DATA" \
    --output_dir "$OUTPUT_PATH" \
    --cache_path "$CACHE_PATH" \
    --overwrite_cache \
    --answer_style full_description \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --save_steps 117 \
    --logging_steps 20 \
    --max_length 1024 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    2>&1 | tee -a "$LOG_PATH"
