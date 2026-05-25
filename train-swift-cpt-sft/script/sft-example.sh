#!/bin/bash
set +u
source /home/xiaoxinyu/.bashrc
conda activate /home/xiaoxinyu/miniconda3/envs/pt

# ---------------- 环境与显卡配置 ----------------
export CUDA_VISIBLE_DEVICES=1,2,3
export NPROC_PER_NODE=3
export MASTER_PORT=$((29500 + RANDOM % 1000))

# 内存优化，防止 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ---------------- ★ 自定义基因模块配置 ★ ----------------
# [必须修改] 请确保这个路径指向你真实的 vocab.json
export GENE_VOCAB_PATH="/data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json"

export FREEZE_NICHEFORMER=ture
export FREEZE_QFORMER_PROJECT=false

# ---------------- 路径配置 ----------------
BASE_MODEL="/data2/xiaoxinyu/project/model_cpt_v6_qformer"
DATA_DIR="/data2/xiaoxinyu/project/data"
OUTPUT_PATH="/data2/xiaoxinyu/project/pretrain-gene/sft_output_v6/gene_$(date +%m%d_%H%M)"
LOG_PATH="/data2/xiaoxinyu/project/pretrain-gene/logs/gene_v6_sft_$(date +%m%d_%H%M).log"

# 数据集路径
# TRAIN_DATA="/data1/xiaoxinyu/project/data/format_dataset/DLPFC_tri_QA_balanced_train_v5.jsonl"
TRAIN_DATA="/data2/xiaoxinyu/project/new_data/gene_data.jsonl"

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
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --freeze_vit true \
    --freeze_aligner false \
    --eval_steps 1000 \
    --save_steps 431 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "$OUTPUT_PATH" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file false \
    --target_modules \
        gene_projector.proj.1 \
        gene_projector.proj.4 \
        gene_qformer.gene_kv_proj.1 \
        gene_qformer.blocks.0.ffn.0 \
        gene_qformer.blocks.0.ffn.3 \
        gene_qformer.blocks.1.ffn.0 \
        gene_qformer.blocks.1.ffn.3 \
        gene_qformer.blocks.2.ffn.0 \
        gene_qformer.blocks.2.ffn.3 \
        gene_qformer.blocks.3.ffn.0 \
        gene_qformer.blocks.3.ffn.3 \
        q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    --deepspeed zero2 2>&1 | tee -a "$LOG_PATH"