#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-ASR vLLM 离线 batch 推理 — 加载一次模型，顺序处理多个 jsonl 分片。

用法:
    python qwen3_asr_batch.py \
        --input_jsonl seg_00.jsonl seg_01.jsonl ... \
        --output_dir /path/to/output \
        --model /path/to/Qwen3-ASR-1.7B \
        --batch_size 128
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_LANG_TAG_RE = re.compile(r"^<\|[a-z_]+\|>\s*")


def strip_lang_tag(text: str) -> str:
    return _LANG_TAG_RE.sub("", text)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] skip bad json: {e}", file=sys.stderr)
    return records


def load_done_keys(output_path: str, audio_field: str,
                   output_field: str) -> set:
    done = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            out = obj.get(output_field, "")
            err = obj.get("error", "")
            if isinstance(out, str) and out.strip() and not err:
                k = obj.get(audio_field, "")
                if k:
                    done.add(k)
    return done


def process_shard(asr, records, audio_field, output_field, batch_size):
    """Run transcribe on a list of records, mutating them in place."""
    todo = []
    for rec in records:
        audio = rec.get(audio_field, "")
        if not audio:
            rec[output_field] = ""
            rec["error"] = f"missing_{audio_field}"
            continue
        todo.append(rec)

    if not todo:
        return 0

    t0 = time.time()
    processed = 0

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        audio_paths = [r[audio_field] for r in batch]

        try:
            results = asr.transcribe(
                audio=audio_paths,
                language=None,
                return_time_stamps=False,
            )
            for rec, res in zip(batch, results):
                text = strip_lang_tag(res.text) if res.text else ""
                rec[output_field] = text
                rec.pop("error", None)
        except Exception as e:
            print(f"  [error] batch at {i}: {e}", file=sys.stderr)
            for rec in batch:
                rec[output_field] = ""
                rec["error"] = f"batch_error: {e}"

        processed += len(batch)
        elapsed = time.time() - t0
        speed = processed / max(elapsed, 0.01)
        eta = (len(todo) - processed) / max(speed, 0.01)
        print(
            f"  [{processed}/{len(todo)}] "
            f"{speed:.1f} items/s, ETA {eta/60:.1f}min",
            file=sys.stderr, flush=True,
        )

    elapsed = time.time() - t0
    print(
        f"  Shard done: {processed} items in {elapsed:.1f}s "
        f"({processed/max(elapsed,0.01):.1f} items/s)",
        file=sys.stderr, flush=True,
    )
    return processed


def main():
    ap = argparse.ArgumentParser(
        description="Qwen3-ASR vLLM batch inference (multi-shard)")
    ap.add_argument("--input_jsonl", "-i", nargs="+", required=True,
                    help="输入 jsonl 文件 (支持多个)")
    ap.add_argument("--output_dir", "-o", required=True,
                    help="输出目录 (每个输入生成 {stem}.asr.jsonl)")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--audio_field", default="audio_path")
    ap.add_argument("--output_field", default="qwen3-omni-asr-result")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--resume", action="store_true",
                    help="跳过已有成功结果的分片")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = []
    for p in args.input_jsonl:
        pp = Path(p)
        if pp.is_file():
            input_files.append(pp)
        else:
            print(f"[warn] not found: {p}", file=sys.stderr)
    if not input_files:
        print("[error] no input files", file=sys.stderr)
        sys.exit(1)

    files_to_process = []
    for in_path in input_files:
        out_path = out_dir / f"{in_path.stem}.asr.jsonl"
        if args.resume and out_path.exists():
            in_lines = sum(1 for _ in open(in_path, "rb"))
            out_lines = sum(1 for _ in open(out_path, "rb"))
            if out_lines >= in_lines - 10:
                print(f"[skip] {in_path.name}: done ({out_lines}/{in_lines})",
                      file=sys.stderr)
                continue
        files_to_process.append((in_path, out_path))

    if not files_to_process:
        print("[*] All shards already done.", file=sys.stderr)
        return

    print(f"[*] Files to process: {len(files_to_process)} / {len(input_files)}",
          file=sys.stderr)

    from qwen_asr import Qwen3ASRModel

    print(f"[*] Loading model: {args.model}", file=sys.stderr, flush=True)
    asr = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print("[*] Model loaded.", file=sys.stderr, flush=True)

    total_processed = 0
    t_global = time.time()

    for in_path, out_path in files_to_process:
        print(f"\n[*] Processing: {in_path.name}", file=sys.stderr, flush=True)
        records = load_jsonl(str(in_path))

        n = process_shard(
            asr, records, args.audio_field, args.output_field, args.batch_size)
        total_processed += n

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Output: {out_path}", file=sys.stderr, flush=True)

    elapsed = time.time() - t_global
    print(
        f"\n[*] All done: {total_processed} items, "
        f"{len(files_to_process)} files, {elapsed:.1f}s total "
        f"({total_processed/max(elapsed,0.01):.1f} items/s)",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
