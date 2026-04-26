# MOSS-Music-Data-Pipeline

<p align="center">
  <img src="./assets/MOSS-Music.png" width="58%" alt="MOSS-Music logo" />
</p>


<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">简体中文</a>
</p>

MOSS-Music-Data-Pipeline is a data annotation and processing pipeline for
building large-scale music understanding corpora, including the training data
for [`MOSS-Music`](https://github.com/OpenMOSS/MOSS-Music).

This repository turns raw music files into structured, chat-formatted training
data for music understanding models.

## Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [Running Mode](#running-mode)
- [Project Structure](#project-structure)
- [Output Artifacts](#output-artifacts)
- [Notes](#notes)

## Overview

This repository covers the full workflow from raw music files to
chat-formatted training samples, including:

- audio duration detection;
- MIR feature extraction (chords, beats, key, melody, instruments, etc.);
- song-structure segmentation;
- lyrics ASR;
- metadata merging and cleanup;
- caption and query generation for downstream training data.

The multimodal inference stages are backend-agnostic. As long as you expose
OpenAI-compatible endpoints, you can plug in **Qwen3-Omni**,
**MusicFlamingo / Audio-Flamingo-3**, and instruction LLMs for the later
generation stages.

## Highlights

- **End-to-end pipeline** from raw audio collection to chat-formatted training
  samples.
- **Modular backends** for ALM captioning, ASR, and LLM generation through
  OpenAI-compatible APIs.
- **Parallel execution design** across full-song analysis, structure parsing,
  and segment-level processing.
- **Scalable data production** through JSONL sharding and distributed execution
  on your own HPC / Kubernetes / Ray cluster.

## Pipeline Overview

<p align="center">
  <img src="./assets/music_pipeline.png" width="95%" />
</p>

### Detailed Execution Graph

```text
Raw Audio Directory
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 0: calc_duration.py                                            │
│   Scan audio/video files -> {"audio_path", "duration"}              │
│   OUTPUT: data.jsonl                                                │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼  --- The following 3 steps can run in PARALLEL ---
     │
     ├──> Step 1a: alm_caption_infer.py -> data.alm
     │      Base Caption via ALM / MusicFlamingo / Qwen3-Omni (API)
     │
     ├──> Step 1b: MusicToolsPipeline (Ray, local CPU/GPU)
     │      Full-song MIR: Chordino + BeatNet + Essentia
     │      CPU -> data.music-cpu/results.jsonl
     │      GPU -> data.music-gpu/results.jsonl
     │
     └──> Step 1c: SongFormer/ (local GPU)
            Song structure annotation
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: song_cut.py                                                 │
│   Segment audio by SongFormer structure                             │
│   OUTPUT: data.sf_cut.jsonl + audio_seg/                            │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼  --- The following 2 steps can run in PARALLEL ---
     │
     ├──> Step 3a: asr_infer.py -> ASR lyrics (API)
     │
     └──> Step 3b: MusicToolsPipeline (Ray, local CPU)
            Segment-level key analysis
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Merge & Clean                                               │
│   4a key_asr_merge -> 4b metadata_merge                             │
│   -> 4c asr_cleanup -> 4d organize_metadata                         │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: Generate Training Data (API)                                │
│   5a caption_generate -> 5b query_generate                          │
│   OUTPUT: data.captions.chat.jsonl                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install Python deps + download SongFormer / MusicFM weights (~8 GB)
bash setup.sh

# 2. Copy and edit the local-mode runner
cp examples/run_pipeline_local.sh my_run.sh
vim my_run.sh   # edit DATA_ROOT, WORK_DIR, API URLs
bash my_run.sh
```

## Running Mode

### Local Mode

Recommended entry point: `examples/run_pipeline_local.sh`

All tasks run on the local machine. `MusicToolsPipeline` uses local CPU / GPU
through Ray, and `SongFormer` runs on local GPUs. External inference services
are expected to be deployed separately for ALM captioning, lyrics ASR, and the
final caption / query generation stage.

Typical setup:

- an **ALM caption** endpoint such as Qwen3-Omni, MusicFlamingo, or
  Audio-Flamingo-3;
- a **lyrics ASR** endpoint such as Qwen3-Omni;
- an **instruction LLM** endpoint for Step 5 generation.

```bash
vllm serve Qwen3-Omni-30B-A3B-Instruct --port 10008
vllm serve Qwen3-Omni-30B-A3B-Instruct --port 8000
vllm serve Qwen3-235B-A22B-Instruct-2507 -tp 8 --port 8001
```

## Project Structure

```text
MOSS-Music-Data-Pipeline/
├── README.md
├── README_zh.md
├── setup.sh
├── requirements.txt
├── patch_beatnet.py
├── .env.example
├── scripts/
│   ├── calc_duration.py
│   ├── alm_caption_infer.py
│   ├── song_cut.py
│   ├── asr_infer.py
│   ├── key_asr_merge.py
│   ├── metadata_merge.py
│   ├── asr_cleanup.py
│   ├── organize_metadata.py
│   ├── caption_generate.py
│   ├── query_generate.py
│   ├── shard_jsonl.py
│   └── merge_sharded_results.py
├── examples/
│   ├── run_pipeline_local.sh
│   ├── launch_qwen3_asr_local.sh
│   └── run_asr_parallel.sh
├── MusicToolsPipeline/
└── SongFormer/
```

## Output Artifacts

| Stage | Output | Description |
|---|---|---|
| Step 0 | `data.jsonl` | Raw audio paths and durations |
| Step 1a | `data.alm` | Base captions generated by the ALM |
| Step 1b | `data.music-cpu/results.jsonl` | Full-song MIR features |
| Step 1b | `data.music-gpu/results.jsonl` | Instrument-related features |
| Step 1c | `data.sf.jsonl` | SongFormer structure annotations |
| Step 2 | `data.sf_cut.jsonl` | Segmented clips and metadata |
| Step 3a | `data.sf_cut.asr/*.jsonl` | Segment-level lyrics ASR results |
| Step 3b | `data.sf_cut.music-cpu/results.jsonl` | Segment-level key / chord analysis |
| Step 4 | `data.meta.clean.organized.jsonl` | Merged and cleaned metadata |
| Step 5 | `data.captions.chat.jsonl` | Final chat-formatted training samples |

## Notes

- `examples/run_pipeline_local.sh` is the recommended full-pipeline entry point.
- `scripts/alm_caption_infer.py` implements Step 1a and supports any
  OpenAI-compatible audio-language model endpoint.
- The original `MOSS-Music/data_pipeline/` implementation has been moved here.
- For large-scale corpora, shard the input JSONL with `scripts/shard_jsonl.py`,
  run each shard independently on your own HPC / Kubernetes / Ray cluster, and
  merge the outputs with `scripts/merge_sharded_results.py`.
