#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5a — 利用 LLM（如 Qwen3-235B）根据 metadata 生成最终 music caption。

支持配置：语言比例(中/英)、caption 格式风格(段落/分节)、和声描述模式。
"""

import os
import sys
import json
import argparse
import asyncio
import random
import aiohttp
from aiohttp import ClientTimeout
from tqdm import tqdm

# ─── Prompt pieces ────────────────────────────────────────────────────────────

STYLE_RULE_PARAGRAPH = """"""

STYLE_RULE_SECTIONS = """Formatting rule:
- You MAY use a structured format with short section headers such as:
  **Tempo**, **Instrument:**, **Song Structure & Dynamics:**, **Harmony & Key:**, **Vocals & Lyrics:**, **Mood & Theme:**, ...
- You can provide an overall description of the section before outputting it.
- Keep it readable and concise. Headings can be omitted if not applicable.
- You may vary the headings order slightly for a more natural / less rigid feel, but do not become chaotic.
"""

HARMONY_RULE_DEGREE = """Harmony description rule:
- When describing harmony, you MAY summarize chord movement for a section and describe it using chord-scale degrees / Roman numerals when the provided key and chord progression support it.
- Do not invent harmonic functions or degrees if they are not reasonably supported by the provided information.
"""

HARMONY_RULE_BASIC = """"""

PROMPT_TEMPLATE = """You are a music description model.
You receive structured information for a single music track:

audio_path:
{AUDIO_PATH}

duration (seconds):
{DURATION_SEC}

Preliminary description:
{BASE_MUSIC_CAPTION}

{SEGMENT_SECTION}{LYRICS_SECTION}{BEATS_SECTION}{TEMPO_SECTION}{KEY_SECTION}{INSTRUMENTS_SECTION}Additional info:
{OTHER_METADATA}

{VOCAL_HINT}

Your task: based on ALL the information above, write a single final caption describing this track for a music dataset.

Requirements:
1. Write in natural expression, with consistent and coherent narration.
2. Summarize: overall style/genre, tempo feel (slow / medium / fast, or approximate BPM if given), key or tonal color, instrumentation and production, vocal characteristics (or clearly state if instrumental), structure and dynamics, and emotional mood.
3. Use the preliminary description as a starting point, you can rewrite in your own words.
4. If the preliminary description contradicts other provided data (e.g., instrumentation, genre, mood), trust the preliminary description — it is the most reliable source. When no lyric text appears in any segment, it does NOT necessarily mean the track is instrumental — trust the preliminary description for vocal/instrumental judgment. Never discuss or reveal contradictions between sources.
5. If lyrics text is present, describe themes and you may cite short representative phrases (no invented lyrics). If a "Reference lyrics" section is provided, treat it as the authoritative lyrical source.
6. Mention only attributes explicitly supported by the provided information.
7. If the audio_path clearly reveals identifiable information such as song title, artist name, or album name, incorporate those details naturally. If such info cannot be clearly determined from audio_path, do NOT mention audio_path or speculate.
8. Clearly describe the overall song structure and approximate timestamps (e.g., intro, verses, chorus sections, bridge, instrumental passages, outro), including how arrangement and dynamics develop across sections.
9. Describe harmonic movement in a musically informed way: explain the main key or tonal center, indicate whether modulation occurs (or state that no clear modulation is evident), and summarize chord progressions conceptually (e.g., "a I–IV–V–vi progression" or "shifts from A minor to C major"). Do NOT list every chord change verbatim; instead describe the harmonic character, functional movement, and tonal color.
10. Do NOT mention field names, JSON, tags, labels, data sources, analysis tools, or data pipelines in the output. The following words/phrases are FORBIDDEN in your output: "ASR", "自动语音识别", "语音识别", "transcription", "speech recognition", "No Lyrics", "metadata", "元数据", "Base Caption", "Base Music Caption", "preliminary description", "标签", "tagged with", "依据标签", "the metadata indicates", "despite the description", "contradicts". Never explain how the data was obtained or processed. Never compare, contrast, or discuss differences between data sources. Write as if you are a human listener describing what you hear — present a single coherent description, not an analysis of multiple sources.
11. Keep the tone objective and descriptive, avoiding direct address to the listener.

{LANGUAGE_RULE}
{STYLE_RULE}
{HARMONY_RULE}

Output: only the final caption. Do not discuss or reveal your reasoning process.
"""


# ─── Utils ────────────────────────────────────────────────────────────────────

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def extract_path(record: dict) -> str:
    return record.get("audio_path") or record.get("path") or \
        record.get("file_path") or ""


def extract_duration_sec(record: dict):
    dur = record.get("duration") or record.get("duration_sec")
    if dur is not None:
        try:
            return float(dur)
        except Exception:
            pass
    song_struct = record.get("song_structure_info") or []
    if isinstance(song_struct, list) and song_struct:
        last_seg = song_struct[-1]
        if isinstance(last_seg, dict):
            end_t = last_seg.get("end")
            try:
                return float(end_t) if end_t is not None else None
            except Exception:
                return None
    return None


def _fmt_time(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "N/A"


def _fmt_key(key_obj) -> str:
    if not isinstance(key_obj, dict):
        return "N/A"
    k = key_obj.get("key")
    sc = key_obj.get("scale")
    if k and sc:
        return f"{k} {sc}"
    return str(k) if k else "N/A"


def _normalize_inst_list(active_list):
    out = []
    for a in active_list or []:
        if isinstance(a, str):
            a = a.strip()
            if a:
                out.append(a)
    return sorted(set(out))


_INST_TARGET_BUCKETS = 30
_INST_MIN_BUCKET = 5.0
_INST_MAX_BUCKET = 30.0


def _collect_instrument_spans(record: dict) -> str:
    changes = record.get("instruments_changes") or []
    if not isinstance(changes, list) or not changes:
        return ""

    duration_sec = extract_duration_sec(record)
    normalized = []
    for ch in changes:
        if not isinstance(ch, dict):
            continue
        try:
            t = float(ch.get("time"))
        except Exception:
            continue
        active = _normalize_inst_list(ch.get("active") or [])
        normalized.append((t, active))

    if not normalized:
        return ""
    normalized.sort(key=lambda x: x[0])

    max_t = float(duration_sec) if duration_sec else normalized[-1][0] + 1.0
    bucket_sec = max(_INST_MIN_BUCKET,
                     min(_INST_MAX_BUCKET, max_t / _INST_TARGET_BUCKETS))
    n_buckets = max(1, int(max_t / bucket_sec) + 1)

    buckets: list[set] = [set() for _ in range(n_buckets)]
    for i, (t, active) in enumerate(normalized):
        if not active:
            continue
        end_t = normalized[i + 1][0] if i + 1 < len(normalized) else max_t
        b_start = int(t / bucket_sec)
        b_end = int(end_t / bucket_sec)
        for b in range(b_start, min(b_end + 1, n_buckets)):
            buckets[b].update(active)

    spans: list[tuple[float, float, list[str]]] = []
    for b, inst_set in enumerate(buckets):
        if not inst_set:
            continue
        key = sorted(inst_set)
        start_t = b * bucket_sec
        end_t = min((b + 1) * bucket_sec, max_t)
        if spans and spans[-1][2] == key:
            spans[-1] = (spans[-1][0], end_t, key)
        else:
            spans.append((start_t, end_t, key))

    lines = [
        f"[{_fmt_time(s)} - {_fmt_time(e)}] {', '.join(a)}"
        for s, e, a in spans
    ]
    return "\n".join(lines)


def _slice_chords_by_time(chord_values, start_t: float, end_t: float):
    picked = []
    for it in chord_values or []:
        try:
            ts = float(it.get("timestamp"))
        except Exception:
            continue
        if ts < start_t or ts >= end_t:
            continue
        chord = it.get("chord")
        if not chord:
            continue
        picked.append(str(chord))

    compact = []
    for c in picked:
        if not compact or compact[-1] != c:
            compact.append(c)
    return compact


def build_segment_pack(record: dict) -> str:
    song_struct = record.get("song_structure_info") or []
    skip_labels = set(record.get("_skip_labels") or [])
    chords_obj = record.get("chords") or {}
    chord_values = chords_obj.get("values") or []

    blocks = []
    for seg in song_struct:
        label = seg.get("label") or ""
        if label in skip_labels:
            continue

        seg_name = seg.get("segment") or label or "segment"
        start_t = seg.get("start")
        end_t = seg.get("end")
        key_str = _fmt_key(seg.get("key"))

        asr_text = seg.get("asr_text") or ""
        asr_text = asr_text.strip() if isinstance(asr_text, str) else ""
        asr_line = asr_text if asr_text else "(none)"

        try:
            st, et = float(start_t), float(end_t)
        except Exception:
            st, et = None, None

        chords = _slice_chords_by_time(chord_values, st, et) \
            if (st is not None and et is not None) else []
        chord_line = " ".join(chords) if chords else "N/A"

        block = (
            f"[{seg_name}] [{_fmt_time(start_t)}] [{key_str}]\n"
            f"lyrics:\n{asr_line}\n\n"
            f"chord_progression:\n{chord_line}\n"
            f"[{_fmt_time(end_t)}]\n"
        )
        blocks.append(block)

    return "\n".join(blocks) if blocks else ""


def collect_other_metadata(record: dict) -> dict:
    other_meta = {}
    for k in [
        "duration_sec", "duration", "title", "artist", "genre_top",
        "genres", "genres_all", "name", "artist_name", "music_tags",
        "mtg_genres", "mtg_instruments", "mtg_moodthemes", "mtg_top50",
    ]:
        if k in record:
            other_meta[k] = record[k]
    if "musicinfo" in record:
        other_meta["musicinfo"] = record["musicinfo"]
    return other_meta


def choose_by_ratio(rng: random.Random, weights: dict, default_key: str):
    items = [(k, float(v)) for k, v in (weights or {}).items()
             if v is not None and float(v) >= 0.0]
    total = sum(w for _, w in items)
    if total <= 0:
        return default_key
    r = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if r <= acc:
            return k
    return items[-1][0]


def _compute_vocal_hint(track_json: dict, base_caption: str) -> str:
    """Determine whether ASR was empty and provide guidance to the LLM."""
    song_struct = track_json.get("song_structure_info") or []
    has_any_asr = False
    for seg in song_struct:
        asr = (seg.get("asr_text") or "").strip()
        if asr:
            has_any_asr = True
            break

    lyrics_source = track_json.get("lyrics_source", "")
    lyrics_text = (track_json.get("lyrics_text") or "").strip()

    if lyrics_text or lyrics_source == "plain_text":
        return ("Vocal hint: Reference lyrics are provided below. "
                "This track has vocals. Use the reference lyrics as "
                "the definitive lyrical content.")

    if has_any_asr:
        return ""

    bc_lower = (base_caption or "").lower()
    vocal_keywords = ["vocal", "singer", "singing", "lyric", "rap", "ballad",
                      "verse", "chorus", "hook"]
    base_suggests_vocal = any(kw in bc_lower for kw in vocal_keywords)

    if base_suggests_vocal:
        return ("Vocal hint: No transcribed lyrics are available, but the "
                "preliminary description describes vocal content. This track "
                "most likely has vocals — describe them based on the "
                "preliminary description.")
    return ""


def _build_lyrics_section(track_json: dict) -> str:
    """Build a reference lyrics section for plain_text lyrics."""
    lyrics_source = track_json.get("lyrics_source", "")
    lyrics_text = (track_json.get("lyrics_text") or "").strip()

    if lyrics_source == "plain_text" and lyrics_text:
        truncated = lyrics_text[:2000]
        if len(lyrics_text) > 2000:
            truncated += "\n[...truncated]"
        return (f"\nReference lyrics (authoritative source — "
                f"use these as the definitive lyrical content):\n"
                f"{truncated}\n")
    return ""


def build_prompt(track_json: dict, lang_mode: str, style_mode: str,
                 harmony_mode: str) -> str:
    audio_path = extract_path(track_json)
    duration_sec = extract_duration_sec(track_json)
    base_caption = (
        track_json.get("Base_Music_Caption")
        or track_json.get("ALM_Caption")
        or track_json.get("MusicFlamigo_Caption")
        or track_json.get("MusicFlamingo_Caption")
        or ""
    )
    segment_pack = build_segment_pack(track_json)
    beatnet = track_json.get("beatnet") or {}
    beats_number = beatnet.get("max_beat_number")
    tempo_bpm = beatnet.get("bpm")
    instruments_text = _collect_instrument_spans(track_json)
    other_meta = collect_other_metadata(track_json)
    overall_key = _fmt_key(track_json.get("music_key"))

    vocal_hint = _compute_vocal_hint(track_json, base_caption)
    lyrics_section = _build_lyrics_section(track_json)

    segment_section = (
        f"Segmented structure + per-segment key + lyrics + per-segment chord progression:\n"
        f"{segment_pack}\n\n"
    ) if segment_pack else ""

    beats_section = f"beats_number:\n{beats_number}\n\n" if beats_number is not None else ""
    tempo_section = f"tempo (BPM):\n{tempo_bpm}\n\n" if tempo_bpm is not None else ""

    key_section = f"overall key:\n{overall_key}\n\n" if overall_key else ""

    instruments_section = f"instruments:\n{instruments_text}\n\n" if instruments_text else ""

    if lang_mode == "en":
        language_rule = (
            "Language rule:\n"
            "- Write the final caption in English.\n"
            "- Do NOT translate or rewrite lyrics text; quote it as-is if needed.\n"
        )
    else:
        language_rule = (
            "Language rule:\n"
            "- Write the final caption in Chinese.\n"
            "- Do NOT translate or rewrite lyrics text; quote it as-is if needed.\n"
        )

    style_rule = STYLE_RULE_SECTIONS if style_mode == "sections" else STYLE_RULE_PARAGRAPH
    harmony_rule = HARMONY_RULE_DEGREE if harmony_mode == "degree" else HARMONY_RULE_BASIC

    return PROMPT_TEMPLATE.format(
        AUDIO_PATH=audio_path or "N/A",
        DURATION_SEC=f"{duration_sec:.6f}" if duration_sec is not None else "N/A",
        BASE_MUSIC_CAPTION=base_caption or "N/A",
        SEGMENT_SECTION=segment_section,
        LYRICS_SECTION=lyrics_section,
        BEATS_SECTION=beats_section,
        TEMPO_SECTION=tempo_section,
        KEY_SECTION=key_section,
        INSTRUMENTS_SECTION=instruments_section,
        OTHER_METADATA=json.dumps(sanitize(other_meta), ensure_ascii=False, indent=2),
        VOCAL_HINT=vocal_hint,
        LANGUAGE_RULE=language_rule.strip(),
        STYLE_RULE=style_rule.strip(),
        HARMONY_RULE=harmony_rule.strip(),
    )


# ─── Path helpers ─────────────────────────────────────────────────────────────

def is_jsonl_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(".jsonl")


def resolve_input_files(input_path: str):
    if os.path.isdir(input_path):
        files = []
        for fname in sorted(os.listdir(input_path)):
            full = os.path.join(input_path, fname)
            if fname.endswith(".jsonl") and os.path.isfile(full):
                files.append((fname, full))
        return files
    if is_jsonl_file(input_path):
        return [(os.path.basename(input_path), input_path)]
    raise ValueError(f"input_path must be a directory or .jsonl: {input_path}")


def resolve_output_targets(output_path: str, input_files):
    if os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        mapping = {n: os.path.join(output_path, n) for n, _ in input_files}
        return "dir", mapping, os.path.join(output_path, "failed_records.log")

    if not os.path.exists(output_path):
        if len(input_files) == 1 and output_path.endswith(".jsonl"):
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            return "single_file", {input_files[0][0]: output_path}, \
                output_path + ".failed.log"
        os.makedirs(output_path, exist_ok=True)
        mapping = {n: os.path.join(output_path, n) for n, _ in input_files}
        return "dir", mapping, os.path.join(output_path, "failed_records.log")

    if len(input_files) == 1 and output_path.endswith(".jsonl"):
        return "single_file", {input_files[0][0]: output_path}, \
            output_path + ".failed.log"

    raise ValueError(f"Multiple inputs require directory output_path: {output_path}")


def read_processed_counts(file_to_output_path: dict):
    processed = {}
    for fname, out_path in file_to_output_path.items():
        if not os.path.exists(out_path):
            processed[fname] = 0
            continue
        c = 0
        with open(out_path, "rb") as f:
            for _ in f:
                c += 1
        processed[fname] = c
    return processed


def count_lines_fast(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


# ─── API call ─────────────────────────────────────────────────────────────────

async def call_llm_async(
    session: aiohttp.ClientSession, api_url: str, api_key: str,
    model_name: str, prompt: str, temperature: float, max_tokens: int,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant with rich music theory knowledge. "
                        "You write high-quality music captions based on structured analysis."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    async with session.post(api_url, headers=headers, json=payload) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
        data = json.loads(text)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected response: {text[:500]}") from e


# ─── Worker ───────────────────────────────────────────────────────────────────

async def worker(
    name: str, session: aiohttp.ClientSession, queue: asyncio.Queue,
    sem: asyncio.Semaphore, out_handles: dict, failed_fh, pbar: tqdm,
    retries: int, rng: random.Random, lang_weights: dict,
    style_weights: dict, harmony_weights: dict, temperature: float,
    max_tokens: int, api_url: str, api_key: str, model_name: str,
    file_to_output_path: dict,
):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        file_name, idx, record = item
        audio_path = extract_path(record)

        lang_mode = choose_by_ratio(rng, lang_weights, default_key="zh")
        if lang_mode not in ("zh", "en"):
            lang_mode = "zh"
        style_mode = choose_by_ratio(rng, style_weights, default_key="paragraph")
        if style_mode not in ("paragraph", "sections"):
            style_mode = "paragraph"
        harmony_mode = choose_by_ratio(rng, harmony_weights, default_key="basic")
        if harmony_mode not in ("basic", "degree"):
            harmony_mode = "basic"

        prompt = build_prompt(record, lang_mode=lang_mode,
                              style_mode=style_mode, harmony_mode=harmony_mode)

        caption = ""
        last_err = ""
        async with sem:
            for attempt in range(retries):
                try:
                    caption = await call_llm_async(
                        session, api_url, api_key, model_name,
                        prompt, temperature=temperature, max_tokens=max_tokens,
                    )
                    break
                except Exception as e:
                    last_err = str(e)
                    if attempt < retries - 1:
                        await asyncio.sleep(2)

        out_path = file_to_output_path[file_name]
        if caption:
            fh = out_handles.get(out_path)
            if fh is None:
                fh = open(out_path, "a", encoding="utf-8")
                out_handles[out_path] = fh
            fh.write(json.dumps({
                "audio_path": audio_path,
                "caption": caption,
                "caption_language": lang_mode,
                "caption_style": style_mode,
                "caption_harmony_mode": harmony_mode,
                "llm_merge_prompt": prompt,
            }, ensure_ascii=False) + "\n")
        else:
            failed_fh.write(json.dumps({
                "file": file_name, "idx": idx, "audio_path": audio_path,
                "error": last_err or "empty caption after retries",
                "caption_language": lang_mode, "caption_style": style_mode,
                "caption_harmony_mode": harmony_mode,
            }, ensure_ascii=False) + "\n")

        pbar.update(1)
        queue.task_done()


# ─── Main async ───────────────────────────────────────────────────────────────

async def main_async(
    input_path: str, output_path: str, concurrency: int, retries: int,
    timeout_sec: int, seed: int, lang_weights: dict, style_weights: dict,
    harmony_weights: dict, temperature: float, max_tokens: int,
    api_url: str, api_key: str, model_name: str, pbar_global: tqdm,
):
    input_files = resolve_input_files(input_path)
    _, file_to_output_path, failed_log_path = \
        resolve_output_targets(output_path, input_files)

    processed_count_by_file = read_processed_counts(file_to_output_path)
    already_processed = sum(processed_count_by_file.values())

    total_lines = sum(count_lines_fast(p) for _, p in input_files)

    print(f"[INFO] {len(input_files)} input file(s), {total_lines} lines, "
          f"already={already_processed}", file=sys.stderr)
    print(f"[INFO] concurrency={concurrency} api_url={api_url}", file=sys.stderr)

    timeout = ClientTimeout(total=timeout_sec)
    connector = aiohttp.TCPConnector(limit=concurrency * 2, ttl_dns_cache=300)

    failed_fh = open(failed_log_path, "a", encoding="utf-8")
    out_handles: dict = {}
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    sem = asyncio.Semaphore(concurrency)
    rngs = [random.Random(seed + i * 10007) for i in range(concurrency)]

    try:
        async with aiohttp.ClientSession(timeout=timeout,
                                         connector=connector) as session:
            worker_tasks = [
                asyncio.create_task(worker(
                    f"w{i}", session, queue, sem, out_handles, failed_fh,
                    pbar_global, retries, rngs[i], lang_weights, style_weights,
                    harmony_weights, temperature, max_tokens, api_url,
                    api_key, model_name, file_to_output_path,
                ))
                for i in range(concurrency)
            ]

            for fname, in_path in input_files:
                skip = processed_count_by_file.get(fname, 0)
                with open(in_path, "r", encoding="utf-8") as f:
                    for _ in range(skip):
                        if not f.readline():
                            break
                    idx = skip
                    for line in f:
                        idx += 1
                        line = line.strip()
                        if not line:
                            pbar_global.update(1)
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            pbar_global.update(1)
                            continue
                        await queue.put((fname, idx, record))

            await queue.join()
            for _ in worker_tasks:
                await queue.put(None)
            await asyncio.gather(*worker_tasks)
    finally:
        for fh in out_handles.values():
            try:
                fh.close()
            except Exception:
                pass
        try:
            failed_fh.close()
        except Exception:
            pass


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_weights(s: str, allowed_keys: set, default_obj: dict):
    if not s:
        return dict(default_obj)
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError
        out = {}
        for k, v in obj.items():
            if k in allowed_keys:
                try:
                    out[k] = float(v)
                except Exception:
                    continue
        return out if out else dict(default_obj)
    except Exception:
        return dict(default_obj)


def parse_args():
    p = argparse.ArgumentParser(
        description="利用 LLM 根据 metadata 生成最终 music caption")
    p.add_argument("input_path", help="输入 jsonl 文件或目录")
    p.add_argument("output_path", help="输出 jsonl 文件或目录")
    p.add_argument("--api_url", type=str, required=True,
                    help="Chat completions API URL（含 /v1/chat/completions）")
    p.add_argument("--model", type=str, required=True, help="LLM 模型名称")
    p.add_argument("--api_key_env", default="INF_API_KEY",
                    help="API Key 环境变量名（默认 INF_API_KEY）")
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout_sec", type=int, default=180)
    p.add_argument("--lang_weights", type=str, default='{"zh":0.5,"en":0.5}')
    p.add_argument("--style_weights", type=str, default='{"paragraph":0.5,"sections":0.5}')
    p.add_argument("--harmony_weights", type=str, default='{"basic":0.4,"degree":0.6}')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=16384)
    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"[FATAL] 环境变量 {args.api_key_env} 未设置", file=sys.stderr)
        sys.exit(1)

    lang_weights = _parse_weights(
        args.lang_weights, {"zh", "en"}, {"zh": 0.5, "en": 0.5})
    style_weights = _parse_weights(
        args.style_weights, {"paragraph", "sections"}, {"paragraph": 0.5, "sections": 0.5})
    harmony_weights = _parse_weights(
        args.harmony_weights, {"basic", "degree"}, {"basic": 0.4, "degree": 0.6})

    async def runner():
        input_files = resolve_input_files(args.input_path)
        _, file_to_output_path, _ = resolve_output_targets(
            args.output_path, input_files)

        processed = read_processed_counts(file_to_output_path)
        already = sum(processed.values())
        total = sum(count_lines_fast(p) for _, p in input_files)
        initial = min(already, total)

        pbar = tqdm(total=total, initial=initial, desc="Caption", unit="line")
        try:
            await main_async(
                input_path=args.input_path, output_path=args.output_path,
                concurrency=args.concurrency, retries=args.retries,
                timeout_sec=args.timeout_sec, seed=args.seed,
                lang_weights=lang_weights, style_weights=style_weights,
                harmony_weights=harmony_weights, temperature=args.temperature,
                max_tokens=args.max_tokens, api_url=args.api_url,
                api_key=api_key, model_name=args.model, pbar_global=pbar,
            )
        finally:
            pbar.close()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
