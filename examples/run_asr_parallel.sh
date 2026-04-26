#!/usr/bin/env bash
# =============================================================================
# 并行 ASR 推理 — 8 分片 × 8 服务器，每个 asr_infer.py 绑定一个 vLLM 实例
#
# 解决瓶颈：vLLM 服务端 librosa 解码音频是 CPU-bound 的，单进程
# round-robin 到 8 个服务器时 GPU 利用率极低。拆成 8 个独立进程后，
# 每个服务器的 CPU 音频解码和 GPU 推理可以充分流水线化。
#
# 用法:
#   bash run_asr_parallel.sh                # 启动 8 路并行
#   bash run_asr_parallel.sh --merge-only   # 仅合并结果 (不重新推理)
# =============================================================================

set -euo pipefail

PIPELINE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="${PIPELINE_ROOT}/scripts"

# ─── 配置 ─────────────────────────────────────────────────────────────────────

INPUT_JSONL="/inspire/qb-ilm/project/embodied-multimodality/public/wxwang/260316_YouTube_Music/Data_process/data.sf_cut.jsonl"
OUTPUT_BASE="/inspire/qb-ilm/project/embodied-multimodality/public/wxwang/260316_YouTube_Music/Data_process/data.sf_cut.asr"
SHARD_DIR="/inspire/qb-ilm/project/embodied-multimodality/public/wxwang/260316_YouTube_Music/Data_process/data.sf_cut.asr_shards"

NUM_WORKERS=8
BASE_PORT=8001
MODEL_NAME="qwen3-asr"

CONCURRENCY=128       # 每个进程的并发数 (不用太高，瓶颈在 CPU 解码)
MAX_TOKENS=2048

# ─── 合并函数 ─────────────────────────────────────────────────────────────────

do_merge() {
    echo "═══ 合并 $NUM_WORKERS 个分片的 ASR 结果 ═══"
    mkdir -p "$OUTPUT_BASE"
    local total=0
    : > "${OUTPUT_BASE}/data.sf_cut.asr.jsonl"
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local shard_out="${SHARD_DIR}/shard_${i}_out"
        for f in "$shard_out"/*.asr.jsonl; do
            [ -f "$f" ] || continue
            local lines
            lines=$(wc -l < "$f")
            total=$((total + lines))
            cat "$f" >> "${OUTPUT_BASE}/data.sf_cut.asr.jsonl"
        done
    done
    echo "[merge] 合并完成: $total 条记录 → ${OUTPUT_BASE}/data.sf_cut.asr.jsonl"
}

if [ "${1:-}" = "--merge-only" ]; then
    do_merge
    exit 0
fi

# ─── Step 1: 分片 ─────────────────────────────────────────────────────────────

echo "═══ Step 1: 将输入分成 $NUM_WORKERS 个分片 ═══"
mkdir -p "$SHARD_DIR"

TOTAL_LINES=$(wc -l < "$INPUT_JSONL")
SHARD_SIZE=$(( (TOTAL_LINES + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "  总行数: $TOTAL_LINES"
echo "  每片:   ~$SHARD_SIZE 行"

python3 "$SCRIPT_DIR/shard_jsonl.py" \
    "$INPUT_JSONL" \
    "$SHARD_DIR" \
    --shard-size "$SHARD_SIZE"

echo ""

# ─── Step 2: 并行启动 asr_infer.py ─────────────────────────────────────────

echo "═══ Step 2: 启动 $NUM_WORKERS 路并行 ASR 推理 ═══"

PIDS=()
SHARD_FILES=($(ls "$SHARD_DIR"/segment_*.jsonl 2>/dev/null | sort))
ACTUAL_SHARDS=${#SHARD_FILES[@]}

if [ "$ACTUAL_SHARDS" -eq 0 ]; then
    echo "错误: 未找到分片文件"
    exit 1
fi

echo "  实际分片数: $ACTUAL_SHARDS"
echo ""

for i in $(seq 0 $((ACTUAL_SHARDS - 1))); do
    SHARD_FILE="${SHARD_FILES[$i]}"
    SERVER_IDX=$((i % NUM_WORKERS))
    PORT=$((BASE_PORT + SERVER_IDX))
    OUT_DIR="${SHARD_DIR}/shard_${i}_out"
    LOG_FILE="${SHARD_DIR}/shard_${i}.log"

    mkdir -p "$OUT_DIR"

    echo "  [shard $i] $(basename "$SHARD_FILE") → http://localhost:$PORT → $OUT_DIR"

    python3 "$SCRIPT_DIR/asr_infer.py" \
        -i "$SHARD_FILE" \
        -o "$OUT_DIR" \
        --servers "http://localhost:$PORT" \
        --model "$MODEL_NAME" \
        --audio_only --strip_lang_tag \
        --api_key_env DUMMY_UNUSED_KEY \
        --concurrency "$CONCURRENCY" \
        --max_tokens "$MAX_TOKENS" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "═══ $ACTUAL_SHARDS 路并行已启动，等待完成... ═══"
echo "  日志: ${SHARD_DIR}/shard_*.log"
echo "  实时查看: tail -f ${SHARD_DIR}/shard_0.log"
echo ""

# ─── Step 3: 等待所有进程 ─────────────────────────────────────────────────────

FAIL=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    if wait "$pid"; then
        echo "  [shard $i] ✓ 完成 (PID $pid)"
    else
        echo "  [shard $i] ✗ 失败 (PID $pid, exit=$?)"
        FAIL=$((FAIL + 1))
    fi
done

echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "⚠️  $FAIL 个分片失败，请检查日志: ${SHARD_DIR}/shard_*.log"
fi

# ─── Step 4: 合并 ────────────────────────────────────────────────────────────

do_merge

echo ""
echo "═══ 并行 ASR 完成 ═══"
