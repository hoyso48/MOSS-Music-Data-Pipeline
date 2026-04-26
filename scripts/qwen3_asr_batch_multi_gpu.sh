#!/usr/bin/env bash
# =============================================================================
# Qwen3-ASR 多 GPU batch 推理 — 在一个节点内 8 张卡并行
#
# 每张卡加载一次模型，顺序处理分配到的所有分片，避免重复加载。
#
# 用法:
#   bash qwen3_asr_batch_multi_gpu.sh <shard_dir> <out_dir> <start_idx> <end_idx>
# =============================================================================

set -euo pipefail

SHARD_DIR="$1"
OUT_DIR="$2"
START_IDX="$3"
END_IDX="$4"

MODEL="/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Qwen3-ASR-1.7B"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BATCH_SIZE=128
MAX_NEW_TOKENS=2048
GPU_MEM=0.9

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "[*] Node GPUs: $NUM_GPUS"
echo "[*] Shards: $START_IDX .. $((END_IDX - 1))"
echo "[*] Model: $MODEL"

mkdir -p "$OUT_DIR"

# 收集需要处理的分片
SHARDS=()
for i in $(seq "$START_IDX" $((END_IDX - 1))); do
    f=$(printf "%s/segment_%02d.jsonl" "$SHARD_DIR" "$i")
    [ -f "$f" ] || f=$(printf "%s/segment_%d.jsonl" "$SHARD_DIR" "$i")
    [ -f "$f" ] || { echo "[warn] shard $i not found, skip"; continue; }
    SHARDS+=("$f")
done

TOTAL=${#SHARDS[@]}
echo "[*] Shards to process: $TOTAL"

if [ "$TOTAL" -eq 0 ]; then
    echo "[*] Nothing to do."
    exit 0
fi

# 按 GPU 分配分片，每个 GPU 用一次 qwen3_asr_batch.py 调用处理所有分配到的文件
PIDS=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_SHARDS=()
    for j in $(seq "$gpu" "$NUM_GPUS" $((TOTAL - 1))); do
        GPU_SHARDS+=("${SHARDS[$j]}")
    done

    [ ${#GPU_SHARDS[@]} -eq 0 ] && continue

    echo "[GPU $gpu] Assigned ${#GPU_SHARDS[@]} shards"

    CUDA_VISIBLE_DEVICES=$gpu python3 "$SCRIPT_DIR/qwen3_asr_batch.py" \
        --input_jsonl "${GPU_SHARDS[@]}" \
        --output_dir "$OUT_DIR" \
        --model "$MODEL" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --gpu_memory_utilization "$GPU_MEM" \
        --resume \
        2>&1 | sed "s/^/[GPU $gpu] /" &

    PIDS+=($!)
done

echo "[*] All $NUM_GPUS GPU workers launched, waiting..."

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

if [ "$FAIL" -gt 0 ]; then
    echo "[!] $FAIL GPU worker(s) had errors"
    exit 1
fi

echo "[*] All done."
