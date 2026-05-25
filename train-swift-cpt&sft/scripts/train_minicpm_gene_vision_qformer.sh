#!/bin/bash
set +u
source /home/xiaoxinyu/.bashrc
conda activate /home/xiaoxinyu/miniconda3/envs/pt

# ---------------- 环境与显卡配置 ----------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MASTER_PORT=$((29500 + RANDOM % 1000))

# 内存优化，防止 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ---------------- ★ 自定义基因模块配置 ★ ----------------
# [必须修改] 请确保这个路径指向你真实的 vocab.json
export GENE_VOCAB_PATH="/data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json"

export FREEZE_NICHEFORMER=false
export FREEZE_QFORMER_PROJECT=false

# ---------------- 路径配置 ----------------
BASE_MODEL="/data2/xiaoxinyu/project/model_cpt_v7_qformer"
DATA_DIR="/data2/xiaoxinyu/project/data"
OUTPUT_PATH="/data2/xiaoxinyu/project/pretrain-gene/sft_output/brain_v7_$(date +%m%d_%H%M)"
LOG_PATH="/data2/xiaoxinyu/project/pretrain-gene/logs/brain_sft_$(date +%m%d_%H%M).log"

# 数据集路径
# TRAIN_DATA="/data1/xiaoxinyu/project/data/format_dataset/DLPFC_tri_QA_balanced_train_v5.jsonl"
TRAIN_DATA="/data1/xiaoxinyu/project/data/sft_DLPFC_tri.jsonl"
# TRAIN_DATA="/data1/xiaoxinyu/project/data/sft_STimage_tri.jsonl"

# 创建日志目录
mkdir -p "$(dirname "$LOG_PATH")"

echo "🚀 Starting Swift SFT for MiniCPM + Gene..." | tee -a "$LOG_PATH"
echo "📂 Model: $BASE_MODEL" | tee -a "$LOG_PATH"
echo "🧬 Gene Vocab: $GENE_VOCAB_PATH" | tee -a "$LOG_PATH"

swift sft \
    --custom_register_path "/data2/xiaoxinyu/project/pretrain-gene/my_custom_model/my_register_qformer.py" \
    --model_type "minicpm_v2_6_gene" \
    --template "minicpm_v2_6_gene" \
    --model "$BASE_MODEL" \
    --dataset "$TRAIN_DATA" \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --freeze_vit false \
    --freeze_aligner false \
    --eval_steps 234 \
    --save_steps 117 \
    --logging_steps 20 \
    --max_length 1024 \
    --output_dir "$OUTPUT_PATH" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file false \
    --deepspeed zero2 2>&1 | tee -a "$LOG_PATH"