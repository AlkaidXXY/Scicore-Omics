#!/bin/bash
set +u
source YOUR_SOURCE_PATH
conda activate YOUR_CONDA_ENV

# ---------------- 环境与显卡配置 ----------------
export CUDA_VISIBLE_DEVICES=1,2,3
export NPROC_PER_NODE=3
export MASTER_PORT=$((29500 + RANDOM % 1000))

# 内存优化，防止 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ---------------- ★ 自定义基因模块配置 ★ ----------------
# [必须修改] 请确保这个路径指向你真实的 vocab.json
export GENE_VOCAB_PATH="/model/gene_tokenizer/vocab.json"

export FREEZE_NICHEFORMER=true
export FREEZE_QFORMER_PROJECT=false

# ---------------- 路径配置 ----------------
BASE_MODEL="/model"
DATA_DIR="YOUR_DATA_DIR"  # 替换为你的数据目录路径
OUTPUT_PATH="YOUR_OUTPUT_DIR"  # 替换为你想保存模型的目录路径
LOG_PATH="YOUR_LOG_DIR/cpt_sft_$(date +%Y%m%d_%H%M%S).log"  # 替换为你想保存日志的目录路径
# 数据集路径

TRAIN_DATA="YOUR_TRAIN_DATA_PATH"  # 替换为你的训练数据路径

# 创建日志目录
mkdir -p "$(dirname "$LOG_PATH")"

echo "🚀 Starting Swift SFT for MiniCPM + Gene..." | tee -a "$LOG_PATH"
echo "📂 Model: $BASE_MODEL" | tee -a "$LOG_PATH"
echo "🧬 Gene Vocab: $GENE_VOCAB_PATH" | tee -a "$LOG_PATH"

swift sft \
    --custom_register_path "/register/my_register_qformer.py" \
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
