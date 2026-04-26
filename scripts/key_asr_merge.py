#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4a — 将裁剪段的 ASR 结果与 Key Change 结果合并为 song_structure_info。

按 orig_audio_path 聚合各段信息，输出每首歌一行。
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, Any

NO_LYRICS_TOKENS = {
    "NO_LYRICS_DETECTED", "NO_LYRIC_DETECTED",
    "NO ASR RESULT", "NO_ASR_RESULT",
    "[inaudible]", "[INAUDIBLE]", "inaudible", "INAUDIBLE", "[Inaudible]",
}


def normalize_asr_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    if not x or x in NO_LYRICS_TOKENS:
        return ""
    return x


def main():
    ap = argparse.ArgumentParser(
        description="合并裁剪段的 Key 和 ASR 结果为 song_structure_info")
    ap.add_argument("--segments_jsonl", required=True,
                    help="裁剪产出的 segments jsonl（含 orig_audio_path / segment）")
    ap.add_argument("--key_results_jsonl", required=True,
                    help="Key 分析产出的 results.jsonl（含 music_cpu.key）")
    ap.add_argument("--asr_jsonl", required=True,
                    help="ASR 结果 jsonl（含 qwen3-omni-asr-result）")
    ap.add_argument("--out_jsonl", required=True,
                    help="输出 jsonl，每首歌一行，含 song_structure_info")
    ap.add_argument("--drop_labels", default="silence",
                    help="不输出的 label（逗号分隔）")
    args = ap.parse_args()

    drop = {x.strip() for x in args.drop_labels.split(",") if x.strip()}

    def iter_jsonl(path: str):
        bad = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
        if bad:
            print(f"[WARN] {path}: 跳过 {bad} 行无法解析的 JSON")

    seg_meta_by_cut: Dict[str, Dict[str, Any]] = {}
    for o in iter_jsonl(args.segments_jsonl):
        seg_meta_by_cut[o["audio_path"]] = o

    asr_by_cut: Dict[str, str] = {}
    for o in iter_jsonl(args.asr_jsonl):
        asr_by_cut[o["audio_path"]] = normalize_asr_text(
            o.get("qwen3-omni-asr-result", ""))

    per_song: Dict[str, list] = defaultdict(list)
    missing_key_join = missing_asr_join = 0

    for r in iter_jsonl(args.key_results_jsonl):
        cut_path = r["audio_path"]

        meta = seg_meta_by_cut.get(cut_path)
        if meta is None:
            missing_key_join += 1
            continue

        seg = meta["segment"]
        if seg["label"] in drop:
            continue

        key_obj = None
        mc = r.get("music_cpu", {})
        if isinstance(mc, dict):
            key_obj = mc.get("key", None)

        asr_text = asr_by_cut.get(cut_path, "")
        if cut_path not in asr_by_cut:
            missing_asr_join += 1

        per_song[meta["orig_audio_path"]].append({
            "segment": seg["label_numbered"],
            "label": seg["label"],
            "idx": seg["label_idx"],
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "key": key_obj,
            "cut_audio_path": cut_path,
            "asr_text": asr_text,
        })

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for orig_audio_path, segs in per_song.items():
            segs.sort(key=lambda x: x["start"])
            fout.write(json.dumps({
                "audio_path": orig_audio_path,
                "song_structure_info": segs,
            }, ensure_ascii=False) + "\n")

    if missing_key_join:
        print(f"[WARN] {missing_key_join} key results 无法匹配 (cut audio_path mismatch)")
    if missing_asr_join:
        print(f"[WARN] {missing_asr_join} segments 无 ASR 结果，回退为空 asr_text")


if __name__ == "__main__":
    main()
