#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/xiaoxinyu/RL/ref_version_v2"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
CKPT_DIR="$ROOT/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

MODEL_PATH="/data2/xiaoxinyu/project/model_cpt_v5_qformer"
EXAMPLE_JSON="$ROOT/example.json"
# EXAMPLE_JSON="/data2/xiaoxinyu/RL/data/pathvqa_open_modelneg_test5.json"
GENE_VOCAB_FILE="/data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json"

PYTHON_BIN="/home/xiaoxinyu/miniconda3/envs/pt/bin/python"

# ===== GPU 配置 =====
# 推荐：train / gen_worker / ref_server 不要重叠
TRAIN_GPU="4,5"
GEN_GPU="3"
REF_GPU="6"
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPU" | awk -F',' '{print NF}')

# ===== 端口 =====
REF_PORT=59875
MASTER_PORT=29611

# ===== 训练配置 =====
ALL_STEPS=1000
LR=5e-7
CLIP_PARAM=0.5
BETA=0.02
ROLLOUT_ACCUM_STEPS=1
GEN_UPDATE_STEPS=8
SAVE_STEPS=100
TRAIN_BATCH_SIZE=1
GRAD_ACC_STEPS=1

# ===== gen_worker 兼容参数 =====
# 注意：固定 output 上传版里，这些“生成参数”基本只为兼容 finetune_gspo.py 保留
GEN_Q_BATCH_SIZE=1
GEN_MAX_NEW_TOKENS=64
GEN_TEMPERATURE=0.3
GEN_TOP_P=0.85
GEN_MAX_SLICE_NUMS=1

export CUDA_HOME=/gpfs/sit_test/apps/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export GENE_VOCAB_FILE="$GENE_VOCAB_FILE"

REF_LOG="$LOG_DIR/ref_server.log"
TRAIN_LOG="$LOG_DIR/train.log"

cleanup() {
    echo ""
    echo "[CLEANUP] Stopping processes..."
    pkill -f "$ROOT/ref_server.py" 2>/dev/null || true
    pkill -f "$ROOT/finetune_gspo.py" 2>/dev/null || true
    pkill -f "$ROOT/gen_worker.py" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[INFO] Cleaning ports..."
fuser -k "${REF_PORT}/tcp" 2>/dev/null || true
fuser -k "${MASTER_PORT}/tcp" 2>/dev/null || true

if [[ ! -f "$EXAMPLE_JSON" ]]; then
    echo "[ERROR] example.json not found: $EXAMPLE_JSON"
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] model path not found: $MODEL_PATH"
    exit 1
fi

echo "========================================"
echo "[INFO] ROOT                 = $ROOT"
echo "[INFO] MODEL_PATH           = $MODEL_PATH"
echo "[INFO] EXAMPLE_JSON         = $EXAMPLE_JSON"
echo "[INFO] GENE_VOCAB_FILE      = $GENE_VOCAB_FILE"
echo "[INFO] PYTHON_BIN           = $PYTHON_BIN"
echo "[INFO] REF_GPU              = $REF_GPU"
echo "[INFO] TRAIN_GPU            = $TRAIN_GPU"
echo "[INFO] GEN_GPU              = $GEN_GPU"
echo "[INFO] NUM_TRAIN_GPUS       = $NUM_TRAIN_GPUS"
echo "[INFO] REF_PORT             = $REF_PORT"
echo "[INFO] MASTER_PORT          = $MASTER_PORT"
echo "[INFO] ALL_STEPS            = $ALL_STEPS"
echo "[INFO] LR                   = $LR"
echo "[INFO] ROLLOUT_ACCUM_STEPS  = $ROLLOUT_ACCUM_STEPS"
echo "[INFO] GEN_UPDATE_STEPS     = $GEN_UPDATE_STEPS"
echo "[INFO] GEN_Q_BATCH_SIZE     = $GEN_Q_BATCH_SIZE"
echo "[INFO] GEN_MAX_NEW_TOKENS   = $GEN_MAX_NEW_TOKENS"
echo "[INFO] GEN_MAX_SLICE_NUMS   = $GEN_MAX_SLICE_NUMS"
echo "========================================"

echo "[INFO] Starting ref_server ..."
CUDA_VISIBLE_DEVICES="$REF_GPU" \
"$PYTHON_BIN" "$ROOT/ref_server.py" \
    --model_path "$MODEL_PATH" \
    --example_json "$EXAMPLE_JSON" \
    --port "$REF_PORT" \
    > "$REF_LOG" 2>&1 &

REF_PID=$!
echo "[INFO] ref_server PID = $REF_PID"

sleep 5

if ! kill -0 "$REF_PID" 2>/dev/null; then
    echo "[ERROR] ref_server exited early. Check $REF_LOG"
    exit 1
fi

echo "[INFO] Waiting for ref_server health..."
HEALTH_OK=0
for i in {1..30}; do
    if curl -s "http://127.0.0.1:${REF_PORT}/health" >/dev/null 2>&1; then
        echo "[INFO] ref_server is healthy."
        HEALTH_OK=1
        break
    fi
    sleep 2
done

if [[ "$HEALTH_OK" -ne 1 ]]; then
    echo "[ERROR] ref_server health check failed. Check $REF_LOG"
    exit 1
fi

echo "[INFO] Starting training ..."
echo "[INFO] Note: current gen_worker is fixed-output uploader, not online generator."

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$NUM_TRAIN_GPUS" \
    --master_port="$MASTER_PORT" \
    "$ROOT/finetune_gspo.py" \
    --model_path "$MODEL_PATH" \
    --example_json "$EXAMPLE_JSON" \
    --ref_port "$REF_PORT" \
    --output_dir "$CKPT_DIR" \
    --all_steps "$ALL_STEPS" \
    --lr "$LR" \
    --clip_param "$CLIP_PARAM" \
    --beta "$BETA" \
    --rollout_accum_steps "$ROLLOUT_ACCUM_STEPS" \
    --gen_update_steps "$GEN_UPDATE_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --gen_device "$GEN_GPU" \
    --gen_q_batch_size "$GEN_Q_BATCH_SIZE" \
    --gen_max_new_tokens "$GEN_MAX_NEW_TOKENS" \
    --gen_temperature "$GEN_TEMPERATURE" \
    --gen_top_p "$GEN_TOP_P" \
    --gen_max_slice_nums "$GEN_MAX_SLICE_NUMS" \
    --gene_vocab_file "$GENE_VOCAB_FILE" \
    > "$TRAIN_LOG" 2>&1

echo "[INFO] Training finished."
echo "[INFO] Logs:"
echo "  - $REF_LOG"
echo "  - $TRAIN_LOG"