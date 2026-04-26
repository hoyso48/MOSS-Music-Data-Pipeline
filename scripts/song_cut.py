#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 — 根据 SongFormer 输出的歌曲结构，将整首歌裁剪为段落级音频片段。

输入: 含 songformer_result 的 JSONL
输出: 裁剪后的音频文件 + 段落级 JSONL
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def sanitize_label(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def numbered_label(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cnt: Dict[str, int] = {}
    out = []
    for seg in segments:
        lab = seg["label"]
        cnt[lab] = cnt.get(lab, 0) + 1
        seg2 = dict(seg)
        seg2["label_idx"] = cnt[lab]
        seg2["label_numbered"] = f"{lab}{cnt[lab]}"
        out.append(seg2)
    return out


def ffprobe_duration(audio_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {audio_path}\n{p.stderr}")
    return float(p.stdout.strip())


def run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")


def compute_out_path(
    out_dir: Path, audio_path: str, base: str, labn: str,
    s2: float, e2: float, fmt: str, keep_rel_root: Optional[str],
) -> Path:
    out_name = f"{base}__{sanitize_label(labn)}_{s2:.3f}_{e2:.3f}.{fmt}"
    if keep_rel_root:
        root = Path(keep_rel_root).resolve()
        ap = Path(audio_path).resolve()
        try:
            rel_parent = ap.parent.relative_to(root)
            out_subdir = out_dir / rel_parent
            out_subdir.mkdir(parents=True, exist_ok=True)
            return out_subdir / out_name
        except Exception:
            return out_dir / out_name
    else:
        return out_dir / out_name


def iter_jsonl_lines_from_file(p: Path) -> Iterable[str]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def collect_input_files(in_jsonl: Optional[str], in_dir: Optional[str]) -> List[Path]:
    files: List[Path] = []
    if in_jsonl:
        files.append(Path(in_jsonl))
    if in_dir:
        d = Path(in_dir)
        files.extend(sorted(d.glob("*.jsonl")))
    uniq = []
    seen: set = set()
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq


def count_total_songs(files: List[Path]) -> int:
    total = 0
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def process_one_song(
    line: str, out_dir_str: str, keep_silence: bool, min_dur: float,
    pad: float, fmt: str, sr: Optional[int], mono: bool,
    accurate_seek: bool, ffmpeg_threads: int, keep_rel_root: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(line)
        audio_path = obj["audio_path"]
        info = obj.get("info", "")
        segments = obj.get("songformer_result", [])
        if not segments:
            return ([], None)

        try:
            dur_total = ffprobe_duration(audio_path)
        except Exception:
            dur_total = float(obj.get("duration", 0.0))

        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(audio_path).stem
        segs_num = numbered_label(segments)
        out_objs: List[Dict[str, Any]] = []

        for seg in segs_num:
            label = seg["label"]
            if (not keep_silence) and label == "silence":
                continue

            s = float(seg["start"])
            e = float(seg["end"])
            s2 = clamp(s - pad, 0.0, dur_total)
            e2 = clamp(e + pad, 0.0, dur_total)
            if e2 <= s2 or (e2 - s2) < min_dur:
                continue

            labn = seg["label_numbered"]
            out_path = compute_out_path(
                out_dir=out_dir, audio_path=audio_path, base=base,
                labn=labn, s2=s2, e2=e2, fmt=fmt, keep_rel_root=keep_rel_root,
            )

            if accurate_seek:
                cmd = ["ffmpeg", "-y", "-v", "error", "-threads", str(ffmpeg_threads),
                       "-i", audio_path, "-ss", f"{s2:.6f}", "-to", f"{e2:.6f}"]
            else:
                seg_dur = max(0.0, e2 - s2)
                cmd = ["ffmpeg", "-y", "-v", "error", "-threads", str(ffmpeg_threads),
                       "-ss", f"{s2:.6f}", "-t", f"{seg_dur:.6f}", "-i", audio_path]

            if mono:
                cmd += ["-ac", "1"]
            if sr is not None:
                cmd += ["-ar", str(sr)]
            if fmt == "flac":
                cmd += ["-c:a", "flac"]
            else:
                cmd += ["-c:a", "pcm_s16le"]
            cmd += [str(out_path)]
            run_cmd(cmd)

            out_objs.append({
                "orig_audio_path": audio_path,
                "audio_path": str(out_path),
                "info": info,
                "segment": {
                    "label": label,
                    "label_idx": seg["label_idx"],
                    "label_numbered": labn,
                    "start": s2,
                    "end": e2,
                },
                "duration": (e2 - s2),
            })

        return (out_objs, None)

    except Exception as ex:
        return ([], f"{type(ex).__name__}: {ex}")


def main():
    ap = argparse.ArgumentParser(
        description="根据 SongFormer 结构标注裁剪音频为段落级片段")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in_jsonl", default=None, help="单个输入 jsonl")
    g.add_argument("--in_dir", default=None, help="输入目录（读取所有 *.jsonl）")

    ap.add_argument("--out_dir", required=True, help="裁剪音频输出目录")
    ap.add_argument("--out_jsonl", required=True, help="输出 jsonl 路径")
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--max_pending", type=int, default=800)
    ap.add_argument("--keep_silence", action="store_true")
    ap.add_argument("--min_dur", type=float, default=0.2)
    ap.add_argument("--pad", type=float, default=0.0)
    ap.add_argument("--format", default="flac", choices=["flac", "wav"])
    ap.add_argument("--sr", type=int, default=None)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--accurate_seek", action="store_true")
    ap.add_argument("--ffmpeg_threads", type=int, default=1)
    ap.add_argument("--keep_rel_root", default=None,
                    help="把 audio_path 相对该 root 的目录结构复刻到 out_dir 下")
    ap.add_argument("--error_jsonl", default=None, help="失败条目输出")
    ap.add_argument("--no_pbar", action="store_true")
    args = ap.parse_args()

    files = collect_input_files(args.in_jsonl, args.in_dir)
    if not files:
        raise SystemExit("No input jsonl files found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, args.workers)
    max_pending = max(workers * 2, args.max_pending)
    total_songs = count_total_songs(files)

    use_pbar = (tqdm is not None) and (not args.no_pbar)
    pbar = tqdm(total=total_songs, desc="Songs", unit="song") if use_pbar else None

    done_songs = done_segments = failed = submitted = 0
    err_f = open(args.error_jsonl, "w", encoding="utf-8") if args.error_jsonl else None

    def log_error(line: str, err: str):
        nonlocal failed
        failed += 1
        if err_f:
            err_f.write(json.dumps({"line": line, "error": err},
                                   ensure_ascii=False) + "\n")

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures: set = set()

            def submit_one(line: str):
                nonlocal submitted
                fut = ex.submit(
                    process_one_song, line, str(out_dir), args.keep_silence,
                    args.min_dur, args.pad, args.format, args.sr, args.mono,
                    args.accurate_seek, args.ffmpeg_threads, args.keep_rel_root,
                )
                futures.add((fut, line))
                submitted += 1

            def consume_one(fut, raw_line):
                nonlocal done_songs, done_segments
                out_objs, err = fut.result()
                if err:
                    log_error(raw_line, err)
                else:
                    done_songs += 1
                    done_segments += len(out_objs)
                    for o in out_objs:
                        fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                if pbar:
                    pbar.update(1)

            for fp in files:
                for line in iter_jsonl_lines_from_file(fp):
                    submit_one(line)
                    if len(futures) >= max_pending:
                        for fut, raw_line in list(futures):
                            if fut.done():
                                futures.remove((fut, raw_line))
                                consume_one(fut, raw_line)
                                break
                        else:
                            fut, raw_line = next(iter(futures))
                            consume_one(fut, raw_line)
                            futures.remove((fut, raw_line))

            for fut, raw_line in as_completed([x[0] for x in futures]):
                raw = None
                for f2, l2 in futures:
                    if f2 == fut:
                        raw = l2
                        break
                consume_one(fut, raw or "")

    if pbar:
        pbar.close()
    if err_f:
        err_f.close()

    print(f"[DONE] total={total_songs} submitted={submitted} "
          f"songs_ok={done_songs} segments={done_segments} failed={failed}")


if __name__ == "__main__":
    main()
