#!/usr/bin/env bash
# =============================================================================
# MOSS-Music Data Pipeline - Local Mode
#
# All tasks run locally. Fill in API_URLs for deployed inference services.
# MusicToolsPipeline (Ray) and SongFormer run on local CPU/GPU.
#
# For large-scale corpora we recommend sharding the input jsonl with
# scripts/shard_jsonl.py and running this script in a distributed fashion
# (e.g. one shard per node via your own HPC / k8s / ray cluster launcher).
# =============================================================================

set -euo pipefail

PIPELINE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="${PIPELINE_ROOT}/scripts"
MTP_DIR="${PIPELINE_ROOT}/MusicToolsPipeline"
SF_DIR="${PIPELINE_ROOT}/SongFormer"

# =========================================================================
#                              User Config
# =========================================================================

DATA_ROOT="/path/to/your/audio"
WORK_DIR="/path/to/your/workdir"

# --- Inference Service API URLs ---
# Deploy these services first, then fill in the URLs:
#   ALM caption model: any OpenAI-compatible ALM service, e.g.
#                        vllm serve Qwen3-Omni-30B-A3B-Instruct --port 10008
#                        vllm serve nvidia/audio-flamingo-3   --port 10008
#   Qwen3-ASR:         vllm/sglang serve Qwen3-Omni --port 8000
#                      or multi-instance: bash examples/launch_qwen3_asr_local.sh
#   LLM:               vllm serve Qwen3-235B-A22B-Instruct -tp 8 --port 8001

ALM_SERVERS=("http://localhost:10008")
ALM_MODEL="Qwen3-Omni-30B-A3B-Instruct"
ASR_SERVERS=("http://localhost:8000")
ASR_MODEL="Qwen3-Omni-30B-A3B-Instruct"
LLM_API_URL="http://localhost:8001/v1/chat/completions"
LLM_MODEL="Qwen3-235B-A22B-Instruct-2507"

export INF_API_KEY="${INF_API_KEY:-}"

# --- Runtime params ---
WORKERS=96
CONCURRENCY=2048
SF_GPUS=8
SF_THREADS_PER_GPU=4
MTP_NUM_WORKERS=35

# =========================================================================
#                              Pipeline
# =========================================================================

mkdir -p "$WORK_DIR"

echo "==========================================================="
echo "  Music Data Process Pipeline - Local Mode"
echo "==========================================================="
echo "  Data root: $DATA_ROOT"
echo "  Work dir:  $WORK_DIR"
echo "==========================================================="
echo ""

wait_with_name() {
    local pid=$1 name=$2
    if wait "$pid"; then
        echo "  OK: $name (PID $pid)"
    else
        echo "  FAIL: $name (PID $pid, exit=$?)"
        echo "    continuing..."
    fi
}

# =============================================================================
# Step 0: Scan audio, get durations
# =============================================================================
echo "=== Step 0: calc_duration ==="
python "$SCRIPT_DIR/calc_duration.py" \
    --root "$DATA_ROOT" \
    --out "$WORK_DIR/data.jsonl" \
    --fail-log "$WORK_DIR/calc_failures.log" \
    --jobs "$WORKERS" \
    --resume --append

TOTAL=$(wc -l < "$WORK_DIR/data.jsonl")
echo "[Step 0] Done: $TOTAL records"
echo ""

# =============================================================================
# Step 1: Parallel feature extraction (1a / 1b / 1c run concurrently)
# =============================================================================

# --- Step 1a: ALM caption model (background) ---
echo "=== Step 1a: ALM caption model (background) ==="
python "$SCRIPT_DIR/alm_caption_infer.py" \
    -i "$WORK_DIR/data.jsonl" \
    -o "$WORK_DIR/data.alm" \
    --servers "${ALM_SERVERS[@]}" \
    --model "$ALM_MODEL" \
    --concurrency "$CONCURRENCY" &
PID_ALM=$!

# --- Step 1b: MusicToolsPipeline local Ray (background) ---
echo "=== Step 1b: Music Tools - CPU pipeline (background) ==="
(
    cd "$MTP_DIR"
    python ray_inference.py \
        --cfg data_path="$WORK_DIR/data.jsonl" \
        --cfg model_path=dummy \
        --cfg output_path="$WORK_DIR/data.music-cpu" \
        --cfg model_type=music_cpu_pipeline \
        --cfg num_workers="$MTP_NUM_WORKERS" \
        --cfg batch_size=4 \
        --cfg num_dataloader_workers=4 \
        --cfg dataloader_type=jsonl
) &
PID_MTP_CPU=$!

echo "=== Step 1b: Music Tools - GPU instrument (background) ==="
(
    cd "$MTP_DIR"
    python ray_inference.py \
        --cfg data_path="$WORK_DIR/data.jsonl" \
        --cfg model_path=dummy \
        --cfg output_path="$WORK_DIR/data.music-gpu" \
        --cfg model_type=essentia_instrument \
        --cfg num_workers=24 \
        --cfg batch_size=4 \
        --cfg num_dataloader_workers=4 \
        --cfg dataloader_type=jsonl
) &
PID_MTP_GPU=$!

# --- Step 1c: SongFormer local GPU (background) ---
echo "=== Step 1c: SongFormer (background) ==="
(
    cd "$SF_DIR"
    python infer_jsonl.py \
        --input_jsonl "$WORK_DIR/data.jsonl" \
        --output_jsonl "$WORK_DIR/data.sf.jsonl" \
        --audio_key audio_path \
        -o "$WORK_DIR/songformer_cache" \
        -gn "$SF_GPUS" \
        -tn "$SF_THREADS_PER_GPU" \
        --model SongFormer \
        --checkpoint SongFormer.safetensors \
        --config_path SongFormer.yaml \
        --resume_merge
) &
PID_SF=$!

echo ""
echo "[Waiting] Step 1 background tasks..."
wait_with_name $PID_ALM     "Step 1a ALM caption"
wait_with_name $PID_SF      "Step 1c SongFormer"
wait_with_name $PID_MTP_CPU "Step 1b Music CPU"
wait_with_name $PID_MTP_GPU "Step 1b Music GPU"
echo ""

# =============================================================================
# Step 2: Segment audio by SongFormer structure
# =============================================================================
echo "=== Step 2: song_cut ==="
python "$SCRIPT_DIR/song_cut.py" \
    --in_jsonl "$WORK_DIR/data.sf.jsonl" \
    --out_dir "$WORK_DIR/audio_seg" \
    --out_jsonl "$WORK_DIR/data.sf_cut.jsonl" \
    --workers "$WORKERS" \
    --keep_rel_root "$DATA_ROOT"

echo ""

# =============================================================================
# Step 3: ASR + Key analysis (can run in parallel)
# =============================================================================

echo "=== Step 3a: ASR lyrics recognition (background) ==="
python "$SCRIPT_DIR/asr_infer.py" \
    -i "$WORK_DIR/data.sf_cut.jsonl" \
    -o "$WORK_DIR/data.sf_cut.asr" \
    --servers "${ASR_SERVERS[@]}" \
    --model "$ASR_MODEL" \
    --concurrency "$CONCURRENCY" &
PID_ASR=$!

echo "=== Step 3b: Key analysis via MusicToolsPipeline (background) ==="
(
    cd "$MTP_DIR"
    python ray_inference.py \
        --cfg data_path="$WORK_DIR/data.sf_cut.jsonl" \
        --cfg model_path=dummy \
        --cfg output_path="$WORK_DIR/data.sf_cut.music-cpu" \
        --cfg model_type=music_cpu_lite_pipeline \
        --cfg num_workers="$MTP_NUM_WORKERS" \
        --cfg batch_size=4 \
        --cfg num_dataloader_workers=4 \
        --cfg dataloader_type=jsonl
) &
PID_KEY=$!

wait_with_name $PID_ASR "Step 3a ASR"
wait_with_name $PID_KEY "Step 3b Key analysis"
echo ""

# =============================================================================
# Step 4: Merge & clean metadata
# =============================================================================
echo "=== Step 4a: key_asr_merge ==="
if [ -s "$WORK_DIR/data.kc.asr.ssi.jsonl" ]; then
    echo "  ⏭ Already exists, skipping"
else
    python "$SCRIPT_DIR/key_asr_merge.py" \
        --segments_jsonl "$WORK_DIR/data.sf_cut.jsonl" \
        --key_results_jsonl "$WORK_DIR/data.sf_cut.music-cpu/results.jsonl" \
        --asr_jsonl "$WORK_DIR/data.sf_cut.asr/data.sf_cut.asr.jsonl" \
        --out_jsonl "$WORK_DIR/data.kc.asr.ssi.jsonl"
fi

echo "=== Step 4b: metadata_merge ==="
if [ -s "$WORK_DIR/data.meta.jsonl" ]; then
    echo "  ⏭ Already exists, skipping"
else
    python "$SCRIPT_DIR/metadata_merge.py" \
        --inputs \
            "$WORK_DIR/data.kc.asr.ssi.jsonl" \
            "$WORK_DIR/data.music-gpu/results.jsonl" \
            "$WORK_DIR/data.music-cpu/results.jsonl" \
            "$WORK_DIR/data.alm" \
            "$WORK_DIR/data.sf.jsonl" \
        --output "$WORK_DIR/data.meta.jsonl"
fi

echo "=== Step 4c: asr_cleanup ==="
if [ -s "$WORK_DIR/data.meta.clean.jsonl" ]; then
    echo "  ⏭ Already exists, skipping"
else
    python "$SCRIPT_DIR/asr_cleanup.py" \
        --in_jsonl "$WORK_DIR/data.meta.jsonl" \
        --out_clean_jsonl "$WORK_DIR/data.meta.clean.jsonl" \
        --out_bad_rows_jsonl "$WORK_DIR/data.meta.bad_rows.jsonl" \
        --out_bad_segments_jsonl "$WORK_DIR/data.meta.bad_seg.jsonl" \
        --workers "$WORKERS"
fi

echo "=== Step 4d: organize_metadata ==="
if [ -s "$WORK_DIR/data.meta.clean.organized.jsonl" ]; then
    echo "  ⏭ Already exists, skipping"
else
    python "$SCRIPT_DIR/organize_metadata.py" \
        --in "$WORK_DIR/data.meta.clean.jsonl" \
        --out "$WORK_DIR/data.meta.clean.organized.jsonl"
fi

echo ""

# =============================================================================
# Step 5: Generate training data
# =============================================================================
echo "=== Step 5a: caption_generate ==="
python "$SCRIPT_DIR/caption_generate.py" \
    "$WORK_DIR/data.meta.clean.organized.jsonl" \
    "$WORK_DIR/data.captions.jsonl" \
    --api_url "$LLM_API_URL" \
    --model "$LLM_MODEL" \
    --concurrency "$CONCURRENCY"

echo "=== Step 5b: query_generate ==="
python "$SCRIPT_DIR/query_generate.py" \
    "$WORK_DIR/data.captions.jsonl" \
    "$WORK_DIR/data.captions.chat.jsonl" \
    --api_url "$LLM_API_URL" \
    --model "$LLM_MODEL" \
    --concurrency "$CONCURRENCY"

echo ""
echo "==========================================================="
echo "  Pipeline Done!"
echo "  Final output: $WORK_DIR/data.captions.chat.jsonl"
echo "==========================================================="
