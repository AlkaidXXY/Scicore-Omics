#!/bin/bash

# ---------------- 环境配置 ----------------
# 指定用于推理的显卡 (单卡即可)
export CUDA_VISIBLE_DEVICES=0

# [必须] 基因词表路径 (与训练时保持一致)
export GENE_VOCAB_PATH="/data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json"

# [可选] 如果你的 register 代码里依赖这个变量来决定模型结构，建议保持一致
# (推理时通常设为 false 也没关系，因为是加载权重，但为了防止初始化报错，设为 true 比较稳妥)
export TUNE_NICHEFORMER=true

# ---------------- 路径配置 ----------------
# 你的 Checkpoint 路径
CKPT_DIR="/data2/liyunfei/swift/output/minicpm_gene_1222_1844/v0-20251222-184431/checkpoint-345"

# 应该能看到 image_processing_minicpmv.py 了
# 自定义注册文件路径
CUSTOM_REGISTER="/data2/liyunfei/swift_project/my_custom_model/my_register.py"

# [可选] 测试集路径
# 如果你想跑整个测试集的评估，请取消下面这行的注释，并填入真实路径TE
TEST_DATA="/data2/xiaoxinyu/project/data/format_dataset/DLPFC_tri_QA_balanced_train_v5.jsonl"

echo "🚀 Starting Inference from: $CKPT_DIR"

# ---------------- 启动推理 ----------------
# 注意：
# 1. --load_dataset_config true 会尝试读取训练时的参数
# 2. 如果你是全参微调 (Full)，Swift 会自动加载全量权重
# 3. 如果是 LoRA，Swift 会自动加载 LoRA 权重并合并(或挂载)

ARGS=" \
    --custom_register_path $CUSTOM_REGISTER \
    --ckpt_dir $CKPT_DIR \
    --model_type minicpm_v2_6_gene \
    --template minicpm_v2_6_gene \
    --torch_dtype bfloat16 \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --stream true \
"

# 判断是否指定了测试集
if [ -n "$TEST_DATA" ]; then
    echo "📊 Mode: Batch Evaluation on $TEST_DATA"
    swift infer $ARGS --dataset "$TEST_DATA" --val_dataset_sample -1
else
    echo "💬 Mode: Interactive Chat (Type 'exit' to quit)"
    swift infer $ARGS
fi