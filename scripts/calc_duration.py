#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0 — 扫描音频/视频目录，获取每个文件的时长，输出 JSONL。
视频文件会先转为 mp3 再探测时长。

输出格式: {"audio_path": "...", "duration": float, "info": "..."}
"""

import os
import sys
import json
import math
import time
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Tuple, List, Set, Dict

AUDIO_EXTS = {
    ".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".wma",
    ".opus", ".ape", ".alac", ".aiff", ".aif", ".amr", ".mka", ".dts",
}
VIDEO_EXTS = {
    ".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm", ".m4v",
    ".ts", ".mts", ".m2ts", ".vob", ".3gp", ".rm", ".rmvb",
}


def which_or_die(bin_name: str):
    from shutil import which
    if which(bin_name) is None:
        print(f"[FATAL] 找不到 {bin_name}，请先安装并确保在 PATH 中。", file=sys.stderr)
        sys.exit(2)


def run_cmd(cmd: list, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out, err
    return p.returncode, out, err


def ffprobe_duration_seconds(path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    code, out, _ = run_cmd(cmd, timeout=120)
    if code != 0:
        return None
    out = out.strip()
    if not out:
        return None
    try:
        return float(out)
    except ValueError:
        return None


def fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_progress(prefix: str, done: int, total: int, ok: int, fail: int,
                    extra: str, t0: float):
    width = 30
    frac = 0.0 if total == 0 else done / total
    filled = int(width * frac)
    bar = "█" * filled + " " * (width - filled)
    elapsed = time.time() - t0
    speed = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / speed if speed > 0 else float("inf")
    eta_str = "??:??:??" if not math.isfinite(eta) else fmt_hms(eta)
    msg = (
        f"\r{prefix} [{bar}] {done}/{total} ({frac*100:6.2f}%) "
        f"ok={ok} fail={fail} {extra} "
        f"{speed:6.2f}it/s ETA {eta_str}"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def load_done_set(jsonl_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                ap = obj.get("audio_path")
                if ap:
                    done.add(ap)
            except Exception:
                continue
    return done


def scan_plan(root: Path, overwrite_mp3: bool, done_set: Set[str]) -> Dict[str, object]:
    audio_probe: List[str] = []
    video_convert: List[str] = []
    scanned_media = 0
    reused_mp3 = 0
    skipped_done = 0

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            ext = p.suffix.lower()

            if ext in AUDIO_EXTS:
                scanned_media += 1
                ap = str(p)
                if ap in done_set:
                    skipped_done += 1
                    continue
                audio_probe.append(ap)

            elif ext in VIDEO_EXTS:
                scanned_media += 1
                mp3p = p.with_suffix(".mp3")
                final_audio = str(mp3p)

                if final_audio in done_set:
                    skipped_done += 1
                    continue

                if mp3p.exists() and (not overwrite_mp3):
                    reused_mp3 += 1
                    audio_probe.append(final_audio)
                else:
                    video_convert.append(str(p))

    return {
        "audio_probe": audio_probe,
        "video_convert": video_convert,
        "scanned_media": scanned_media,
        "reused_mp3": reused_mp3,
        "skipped_done": skipped_done,
    }


def worker_probe_audio(audio_path: str):
    p = Path(audio_path)
    dur = ffprobe_duration_seconds(str(p))
    if dur is None or math.isnan(dur) or dur <= 0:
        return {"ok": False, "audio_path": audio_path, "fail_reason": "duration_fail"}
    return {"ok": True, "audio_path": audio_path, "duration": dur, "info": p.stem}


def worker_convert_video(args_tuple):
    video_path, bitrate, overwrite_mp3 = args_tuple
    v = Path(video_path)
    mp3_path = v.with_suffix(".mp3")

    if mp3_path.exists() and (not overwrite_mp3):
        return {"ok": True, "status": "reused", "mp3_path": str(mp3_path),
                "video_path": video_path}

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y" if overwrite_mp3 else "-n",
        "-i", str(v), "-vn", "-map", "a:0?",
        "-acodec", "libmp3lame", "-b:a", bitrate,
        str(mp3_path),
    ]
    code, _, err = run_cmd(cmd, timeout=60 * 60 * 6)
    if code != 0:
        return {"ok": False, "video_path": video_path,
                "fail_reason": f"ffmpeg_fail: {err.strip()[:200]}"}

    if mp3_path.exists() and mp3_path.stat().st_size > 0:
        return {"ok": True, "status": "converted", "mp3_path": str(mp3_path),
                "video_path": video_path}

    return {"ok": False, "video_path": video_path,
            "fail_reason": "mp3_missing_after_convert"}


def main():
    parser = argparse.ArgumentParser(
        description="扫描音频/视频目录，探测时长，输出 JSONL。视频先转 mp3 再探测。")
    parser.add_argument("--root", required=True, help="根目录")
    parser.add_argument("--out", default="audio_durations.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--fail-log", default="failures.log", help="失败日志文件")
    parser.add_argument("--jobs", type=int, default=0,
                        help="并行进程数（0=CPU核数）")
    parser.add_argument("--bitrate", default="192k", help="视频转 mp3 码率")
    parser.add_argument("--overwrite-mp3", action="store_true",
                        help="覆盖同名 mp3 重新转码")
    parser.add_argument("--append", action="store_true", help="追加写 out")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：跳过 out 里已完成的（建议配合 --append）")
    parser.add_argument("--progress-every", type=int, default=50,
                        help="每处理N条刷新一次进度")
    args = parser.parse_args()

    which_or_die("ffprobe")
    which_or_die("ffmpeg")

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    fail_log = Path(args.fail_log).resolve()

    if not root.exists():
        print(f"[FATAL] root 不存在: {root}", file=sys.stderr)
        sys.exit(2)

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)

    done_set: Set[str] = set()
    if args.resume:
        done_set = load_done_set(out_path)
        args.append = True
        print(f"[RESUME] loaded_done={len(done_set)} from {out_path}")

    if (not args.append) and out_path.exists():
        out_path.unlink()
    out_mode = "a" if args.append else "w"

    fail_f = open(fail_log, "a", encoding="utf-8")
    out_f = open(out_path, out_mode, encoding="utf-8")

    print(f"[SCAN] walking: {root}")
    plan = scan_plan(root, overwrite_mp3=args.overwrite_mp3, done_set=done_set)
    audio_probe: List[str] = plan["audio_probe"]
    video_convert: List[str] = plan["video_convert"]

    print(f"[PLAN] scanned={plan['scanned_media']} audio_probe={len(audio_probe)} "
          f"video_convert={len(video_convert)} reused_mp3={plan['reused_mp3']} "
          f"skipped_done={plan['skipped_done']} jobs={jobs}")

    if not audio_probe and not video_convert:
        print("[DONE] 没有需要处理的文件。")
        out_f.close()
        fail_f.close()
        return

    seen_write: Set[str] = set(done_set)
    skip_dupwrite = 0
    total_seconds = 0.0
    failed = 0
    ok_written = 0

    # Phase 1: probe audio files
    if audio_probe:
        print("\n[PHASE1] PROBE audio")
        t0 = time.time()
        done = ok = fail = 0
        total = len(audio_probe)
        render_progress("PROBE1", 0, total, 0, 0, "", t0)

        with Pool(processes=jobs) as pool:
            for res in pool.imap_unordered(worker_probe_audio, audio_probe, chunksize=20):
                done += 1
                if res.get("ok"):
                    ap = res["audio_path"]
                    if ap in seen_write:
                        skip_dupwrite += 1
                    else:
                        seen_write.add(ap)
                        ok += 1
                        ok_written += 1
                        total_seconds += float(res["duration"])
                        out_f.write(json.dumps({
                            "audio_path": ap,
                            "duration": res["duration"],
                            "info": res["info"],
                        }, ensure_ascii=False) + "\n")
                else:
                    fail += 1
                    failed += 1
                    fail_f.write(f"{res.get('audio_path')}\t{res.get('fail_reason')}\n")

                if done % args.progress_every == 0 or done == total:
                    render_progress("PROBE1", done, total, ok, fail, "", t0)

        sys.stdout.write("\n")

    # Phase 2: convert videos then probe
    if video_convert:
        print("\n[PHASE2] CONVERT videos")
        t1 = time.time()
        conv_done = conv_ok = conv_fail = converted = 0
        conv_total = len(video_convert)
        render_progress("CONV  ", 0, conv_total, 0, 0, "", t1)

        mp3_after_convert: List[str] = []
        with Pool(processes=jobs) as pool:
            it = pool.imap_unordered(
                worker_convert_video,
                ((vp, args.bitrate, args.overwrite_mp3) for vp in video_convert),
                chunksize=5,
            )
            for res in it:
                conv_done += 1
                if res.get("ok"):
                    conv_ok += 1
                    if res.get("status") == "converted":
                        converted += 1
                    mp3p = res.get("mp3_path")
                    if mp3p:
                        mp3_after_convert.append(mp3p)
                else:
                    conv_fail += 1
                    failed += 1
                    fail_f.write(f"{res.get('video_path')}\t{res.get('fail_reason')}\n")

                if conv_done % args.progress_every == 0 or conv_done == conv_total:
                    render_progress("CONV  ", conv_done, conv_total, conv_ok, conv_fail,
                                    f"converted={converted}", t1)

        sys.stdout.write("\n")

        if mp3_after_convert:
            print("[PHASE2] PROBE converted mp3")
            t2 = time.time()
            done2 = ok2 = fail2 = 0
            total2 = len(mp3_after_convert)
            render_progress("PROBE2", 0, total2, 0, 0, "", t2)

            with Pool(processes=jobs) as pool:
                for res in pool.imap_unordered(worker_probe_audio, mp3_after_convert,
                                               chunksize=20):
                    done2 += 1
                    if res.get("ok"):
                        ap = res["audio_path"]
                        if ap in seen_write:
                            skip_dupwrite += 1
                        else:
                            seen_write.add(ap)
                            ok2 += 1
                            ok_written += 1
                            total_seconds += float(res["duration"])
                            out_f.write(json.dumps({
                                "audio_path": ap,
                                "duration": res["duration"],
                                "info": res["info"],
                            }, ensure_ascii=False) + "\n")
                    else:
                        fail2 += 1
                        failed += 1
                        fail_f.write(f"{res.get('audio_path')}\t{res.get('fail_reason')}\n")

                    if done2 % args.progress_every == 0 or done2 == total2:
                        render_progress("PROBE2", done2, total2, ok2, fail2, "", t2)

            sys.stdout.write("\n")

    out_f.close()
    fail_f.close()

    print(f"\n[DONE] ok_written={ok_written} failed={failed} "
          f"total_duration={fmt_hms(total_seconds)}")


if __name__ == "__main__":
    main()
