#!/usr/bin/env bash
# =============================================================================
# 本地 8-GPU 并行部署 Qwen3-ASR-1.7B (每张卡一个 vLLM 实例)
#
# 用法:
#   bash launch_qwen3_asr_local.sh          # 启动 8 个实例 (端口 8001-8008)
#   bash launch_qwen3_asr_local.sh --stop   # 停止所有实例
#   bash launch_qwen3_asr_local.sh --check  # 检查各实例是否就绪
#
# 模型只有 1.7B，单卡即可跑满，8 张 4090 分 8 个独立实例吞吐量最大。
# =============================================================================

set -euo pipefail

VENV="/inspire/ssd/project/embodied-multimodality/public/ywang/uv/qwen3-asr-vllm/.venv/bin/activate"
MODEL="/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Qwen3-ASR-1.7B"
MODEL_NAME="qwen3-asr"
BASE_PORT=8001
NUM_GPUS=8
PID_DIR="/tmp/qwen3-asr-pids"

do_stop() {
    echo "=== 停止所有 Qwen3-ASR 实例 ==="
    if [ -d "$PID_DIR" ]; then
        for f in "$PID_DIR"/*.pid; do
            [ -f "$f" ] || continue
            local pid
            pid=$(cat "$f")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Killing GPU $(basename "$f" .pid) (PID $pid)"
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$f"
        done
    fi
    echo "Done."
}

do_check() {
    echo "=== 检查 Qwen3-ASR 实例状态 ==="
    local ready=0
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        local port=$((BASE_PORT + i))
        local status
        status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/models" 2>/dev/null || echo "000")
        if [ "$status" = "200" ]; then
            echo "  GPU $i (port $port): ✓ Ready"
            ready=$((ready + 1))
        else
            echo "  GPU $i (port $port): ✗ Not ready (HTTP $status)"
        fi
    done
    echo ""
    echo "$ready / $NUM_GPUS instances ready."
}

do_start() {
    source "$VENV"
    mkdir -p "$PID_DIR"

    echo "=== 启动 $NUM_GPUS 个 Qwen3-ASR vLLM 实例 ==="
    echo "  模型: $MODEL"
    echo "  端口: $BASE_PORT ~ $((BASE_PORT + NUM_GPUS - 1))"
    echo ""

    for i in $(seq 0 $((NUM_GPUS - 1))); do
        local port=$((BASE_PORT + i))

        if [ -f "$PID_DIR/$i.pid" ]; then
            local old_pid
            old_pid=$(cat "$PID_DIR/$i.pid")
            if kill -0 "$old_pid" 2>/dev/null; then
                echo "  GPU $i (port $port): 已在运行 (PID $old_pid), 跳过"
                continue
            fi
        fi

        echo "  GPU $i (port $port): 启动中..."
        CUDA_VISIBLE_DEVICES=$i qwen-asr-serve "$MODEL" \
            --gpu-memory-utilization 0.9 \
            --host 0.0.0.0 \
            --port "$port" \
            --served-model-name "$MODEL_NAME" \
            --allowed-local-media-path / \
            --max-model-len 4096 \
            --max-num-seqs 256 \
            > "/tmp/qwen3-asr-gpu${i}.log" 2>&1 &

        echo $! > "$PID_DIR/$i.pid"
        sleep 1
    done

    echo ""
    echo "=== 全部实例已启动 (后台运行) ==="
    echo ""
    echo "服务地址:"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        echo "  http://localhost:$((BASE_PORT + i))"
    done
    echo ""
    echo "日志: /tmp/qwen3-asr-gpu{0..7}.log"
    echo ""
    echo "等待就绪后可用以下命令检查:"
    echo "  bash $0 --check"
    echo ""
    echo "停止所有实例:"
    echo "  bash $0 --stop"
}

case "${1:-start}" in
    --stop|-s)   do_stop ;;
    --check|-c)  do_check ;;
    start|--start|-S|"") do_start ;;
    *)
        echo "用法: $0 [--start | --stop | --check]"
        exit 1
        ;;
esac
