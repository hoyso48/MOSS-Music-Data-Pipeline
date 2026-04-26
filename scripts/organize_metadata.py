#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4d — 整理 metadata 最终结构。

- 保留/补齐 song_structure_info 中的 asr_text
- 兼容旧格式（key_change + qwen3_omni_asr_result）并转为新格式
- 抽平 music_cpu / music_gpu 为顶层字段
- 删除临时字段
- 重排字段顺序
"""

import argparse
import json
import re
from typing import Optional
from tqdm import tqdm

TS_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2}\.\d{3})\]")
LABEL_RE = re.compile(r"^\[([^\]]+)\]\s*\[(\d{2}):(\d{2}):(\d{2}\.\d{3})\]\s*$")


def ts_to_sec(hh: str, mm: str, ss_ms: str) -> float:
    return int(hh) * 3600 + int(mm) * 60 + float(ss_ms)


def parse_asr_blocks(asr_text: str):
    lines = asr_text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = LABEL_RE.match(line)
        if not m:
            i += 1
            continue

        label = m.group(1).strip()
        start = ts_to_sec(m.group(2), m.group(3), m.group(4))
        i += 1
        text_lines = []
        end = None

        while i < len(lines):
            cur = lines[i].strip()
            if LABEL_RE.match(cur):
                break
            tsm = TS_RE.match(cur)
            if tsm and cur.startswith("[") and cur.endswith("]") \
                    and len(cur) == len(tsm.group(0)):
                end = ts_to_sec(tsm.group(1), tsm.group(2), tsm.group(3))
                i += 1
                break
            if cur:
                text_lines.append(cur)
            i += 1

        blocks.append({
            "label": label,
            "start": float(start),
            "end": float(end) if end is not None else None,
            "text": "\n".join(text_lines).strip(),
        })

    for j in range(len(blocks)):
        if blocks[j]["end"] is None and j + 1 < len(blocks):
            blocks[j]["end"] = float(blocks[j + 1]["start"])

    return blocks


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def align_asr_to_keychange(asr_blocks, key_change, min_overlap=0.02):
    out = []
    for seg in key_change:
        s0 = float(seg.get("start", 0.0))
        s1 = float(seg.get("end", 0.0))
        texts = []
        for blk in asr_blocks:
            b0 = float(blk.get("start", 0.0))
            b1 = blk.get("end")
            if b1 is None:
                continue
            if overlap(s0, s1, b0, float(b1)) >= min_overlap:
                t = (blk.get("text") or "").strip()
                if t:
                    texts.append(t)
        seg2 = dict(seg)
        seg2["asr_text"] = "\n".join(texts).strip()
        out.append(seg2)
    return out


def normalize_song_structure_info(song_structure_info):
    if not isinstance(song_structure_info, list):
        return song_structure_info
    out = []
    for seg in song_structure_info:
        seg2 = dict(seg)
        seg2.setdefault("asr_text", "")
        out.append(seg2)
    return out


def transform_record(obj):
    if isinstance(obj.get("song_structure_info"), list):
        song_structure_info = normalize_song_structure_info(obj["song_structure_info"])
    else:
        key_change = obj.get("key_change")
        asr_text = obj.get("qwen3_omni_asr_result")

        if isinstance(key_change, list):
            if isinstance(asr_text, str) and asr_text.strip():
                asr_blocks = parse_asr_blocks(asr_text)
                song_structure_info = align_asr_to_keychange(asr_blocks, key_change)
            else:
                song_structure_info = [dict(x) for x in key_change]
                for x in song_structure_info:
                    x.setdefault("asr_text", "")
        else:
            song_structure_info = None

    obj.pop("key_change", None)
    obj.pop("qwen3_omni_asr_result", None)
    obj.pop("Qwen3-Omni-I", None)
    obj.pop("_alm_prompt", None)
    obj.pop("_musicflamingo_prompt", None)
    obj.pop("song_structure_info", None)

    music_cpu = obj.pop("music_cpu", None)
    if isinstance(music_cpu, dict):
        if "chords" in music_cpu:
            obj["chords"] = music_cpu["chords"]
        if "beatnet" in music_cpu:
            obj["beatnet"] = music_cpu["beatnet"]
        if "key" in music_cpu:
            obj["music_key"] = music_cpu["key"]

    music_gpu = obj.pop("music_gpu", None)
    if isinstance(music_gpu, dict):
        if "changes" in music_gpu:
            obj["instruments_changes"] = music_gpu["changes"]

    new_obj = {}
    if "audio_path" in obj:
        new_obj["audio_path"] = obj["audio_path"]
    if song_structure_info is not None:
        new_obj["song_structure_info"] = song_structure_info
    for k, v in obj.items():
        if k != "audio_path":
            new_obj[k] = v

    return new_obj


def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser(description="整理 metadata 最终结构")
    parser.add_argument("--in", dest="inp", required=True, help="输入 JSONL")
    parser.add_argument("--out", dest="out", required=True, help="输出 JSONL")
    parser.add_argument("--errors", default=None, help="错误日志输出路径")
    args = parser.parse_args()

    total_lines = count_lines(args.inp)
    err_f = open(args.errors, "w", encoding="utf-8") if args.errors else None

    with open(args.inp, "r", encoding="utf-8") as f_in, \
         open(args.out, "w", encoding="utf-8") as f_out:
        for ln, line in enumerate(tqdm(f_in, total=total_lines, desc="Processing"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj2 = transform_record(obj)
                f_out.write(json.dumps(obj2, ensure_ascii=False) + "\n")
            except Exception as e:
                if err_f:
                    err_f.write(json.dumps({
                        "line": ln, "error": str(e), "raw": line[:1000],
                    }, ensure_ascii=False) + "\n")

    if err_f:
        err_f.close()


if __name__ == "__main__":
    main()
