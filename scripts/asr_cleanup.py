#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4c — 清洗 ASR 结果，检测并修复幻觉文本。

对 song_structure_info 中每个 segment 的 asr_text 进行检查：
- 重复片段 → 压缩为最多 keep_times 次
- 其他幻觉（极端重复字符、速度异常等）→ 置空
"""

import argparse
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from itertools import repeat

from tqdm import tqdm


def unique_char_ratio(text: str) -> float:
    text = text.strip()
    if not text:
        return 1.0
    return len(set(text)) / max(len(text), 1)


def max_char_frequency_ratio(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0
    c = Counter(text)
    return c.most_common(1)[0][1] / len(text)


def has_long_same_char_run(text: str, run: int = 8) -> bool:
    return re.search(rf"(.)\1{{{run},}}", text) is not None


def find_repeated_chunk(text: str, min_chunk: int = 2, max_chunk: int = 200):
    for size in range(min_chunk, max_chunk + 1):
        pattern = rf"(.{{{size}}})\1+"
        m = re.search(pattern, text, flags=re.S)
        if m:
            chunk = m.group(1)
            count = len(m.group(0)) // len(chunk)
            return chunk, count
    return None, 0


def chars_per_second(text: str, duration: float) -> float:
    duration = max(duration, 1e-3)
    return len(text.strip()) / duration


def compress_repetition(text: str, keep_times: int = 3) -> str:
    chunk, count = find_repeated_chunk(text)
    if chunk and count > keep_times:
        return chunk * keep_times
    return text


def is_hallucination(
    text: str, duration: float, *,
    never_hallu_len: int, min_len_for_hallu: int,
    hard_same_char_run: int, cps_threshold: float,
    unique_threshold: float, max_char_ratio_threshold: float,
    same_char_run: int, repeated_chunk_repeats: int,
) -> Tuple[bool, List[str], Dict[str, float]]:
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    t = (text or "").strip()
    if not t:
        return False, reasons, metrics

    L = len(t)
    metrics["len"] = L

    if L < never_hallu_len:
        return False, reasons, metrics

    cps = chars_per_second(t, duration)
    u = unique_char_ratio(t)
    mcr = max_char_frequency_ratio(t)
    metrics.update({"cps": cps, "unique_ratio": u, "max_char_ratio": mcr})

    if has_long_same_char_run(t, run=hard_same_char_run):
        return True, ["hard_same_char_run"], metrics

    if L < min_len_for_hallu:
        return False, reasons, metrics

    if cps > cps_threshold:
        reasons.append("speed")
    if u < unique_threshold:
        reasons.append("low_unique")
    if mcr > max_char_ratio_threshold:
        reasons.append("max_char_ratio")

    chunk, count = find_repeated_chunk(t)
    if chunk and count >= repeated_chunk_repeats:
        reasons.append("repeated_chunk")

    return (len(reasons) > 0), reasons, metrics


def process_line(
    line: str, set_bad_to: str, never_hallu_len: int,
    min_len_for_hallu: int, hard_same_char_run: int,
    cps_threshold: float, unique_threshold: float,
    max_char_ratio_threshold: float, same_char_run: int,
    repeated_chunk_repeats: int,
) -> Tuple[str, Optional[str], List[str], int]:
    line = line.strip()
    if not line:
        return "", None, [], 0

    raw_obj = json.loads(line)
    obj = json.loads(line)
    row_has_bad = False
    bad_seg_lines: List[str] = []
    bad_count = 0

    for i, seg in enumerate(obj.get("song_structure_info", [])):
        text = seg.get("asr_text", "")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        duration = max(end - start, 1e-3)

        bad, reasons, metrics = is_hallucination(
            text, duration,
            never_hallu_len=never_hallu_len,
            min_len_for_hallu=min_len_for_hallu,
            hard_same_char_run=hard_same_char_run,
            cps_threshold=cps_threshold,
            unique_threshold=unique_threshold,
            max_char_ratio_threshold=max_char_ratio_threshold,
            same_char_run=same_char_run,
            repeated_chunk_repeats=repeated_chunk_repeats,
        )

        if bad:
            row_has_bad = True
            bad_count += 1
            if "repeated_chunk" in reasons:
                seg["asr_text"] = compress_repetition(text, keep_times=3)
            else:
                seg["asr_text"] = set_bad_to

            bad_seg_lines.append(json.dumps({
                "audio_path": obj.get("audio_path"),
                "segment_index": i,
                "reasons": reasons,
                "metrics": metrics,
                "original_text": text,
                "fixed_text": seg["asr_text"],
            }, ensure_ascii=False))

    cleaned_line = json.dumps(obj, ensure_ascii=False)
    bad_row_line = json.dumps(raw_obj, ensure_ascii=False) if row_has_bad else None
    return cleaned_line, bad_row_line, bad_seg_lines, bad_count


def main():
    ap = argparse.ArgumentParser(description="清洗 ASR 幻觉文本")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_clean_jsonl", required=True)
    ap.add_argument("--out_bad_rows_jsonl", required=True)
    ap.add_argument("--out_bad_segments_jsonl", required=True)
    ap.add_argument("--set_bad_to", default="")
    ap.add_argument("--never_hallu_len", type=int, default=1000,
                    help="短于此长度永不判为幻觉")
    ap.add_argument("--min_len_for_hallu", type=int, default=200)
    ap.add_argument("--hard_same_char_run", type=int, default=80)
    ap.add_argument("--cps_threshold", type=float, default=12.0)
    ap.add_argument("--unique_threshold", type=float, default=0.10)
    ap.add_argument("--max_char_ratio_threshold", type=float, default=0.55)
    ap.add_argument("--same_char_run", type=int, default=8)
    ap.add_argument("--repeated_chunk_repeats", type=int, default=6)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--chunksize", type=int, default=128)
    args = ap.parse_args()

    mp_context = mp.get_context("spawn")

    with open(args.in_jsonl, "r", encoding="utf-8", errors="ignore") as fin, \
         open(args.out_clean_jsonl, "w", encoding="utf-8") as fclean, \
         open(args.out_bad_rows_jsonl, "w", encoding="utf-8") as fbadrows, \
         open(args.out_bad_segments_jsonl, "w", encoding="utf-8") as fbadsegs:

        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp_context) as ex:
            iterator = ex.map(
                process_line, fin,
                repeat(args.set_bad_to), repeat(args.never_hallu_len),
                repeat(args.min_len_for_hallu), repeat(args.hard_same_char_run),
                repeat(args.cps_threshold), repeat(args.unique_threshold),
                repeat(args.max_char_ratio_threshold), repeat(args.same_char_run),
                repeat(args.repeated_chunk_repeats),
                chunksize=args.chunksize,
            )
            for cleaned_line, bad_row_line, bad_seg_lines, bad_count in tqdm(iterator):
                if cleaned_line:
                    fclean.write(cleaned_line + "\n")
                if bad_row_line:
                    fbadrows.write(bad_row_line + "\n")
                for s in bad_seg_lines:
                    fbadsegs.write(s + "\n")


if __name__ == "__main__":
    main()
