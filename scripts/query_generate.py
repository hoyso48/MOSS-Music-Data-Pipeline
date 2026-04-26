#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5b — 根据 caption 生成训练用 query，输出对话格式数据。

利用 LLM 为每条 caption 生成自然语言 query，
最终输出 {"messages": [...], "audios": [...], "meta": {...}} 格式。
"""

import os
import sys
import json
import glob
import asyncio
import hashlib
import random
import argparse
import aiohttp
from aiohttp import ClientTimeout
from tqdm import tqdm

# ─── Fallback prompts ────────────────────────────────────────────────────────

FALLBACK_PROMPTS = {
    ("paragraph", "zh"): [
        "请描述一下这段音乐。",
        "描述一下这段音乐。",
        "请简单描述这首音乐的整体特点。",
        "请用自然语言介绍一下这段音频。",
        "请简要描述这段音乐的整体听感。",
        "请描述一下这段音乐的整体感觉。",
        "这段音乐听起来怎么样？请描述一下。",
        "请介绍一下这首曲目的整体风格和氛围。",
        "请描述一下这段音乐给你的整体印象。",
        "请详细描述一下这段音乐的整体风格和听觉感受。",
        "请从整体上描述这段音乐给你的感受。",
        "请描述一下这段音乐的风格和特点。",
        "用你的话介绍一下这段音乐。",
        "请概括一下这段音乐的整体特征。",
        "请描述你从这段音乐中听到了什么。",
    ],
    ("paragraph", "en"): [
        "Describe this piece of music.",
        "Give a brief description of this track.",
        "Write a natural description of this audio clip.",
        "Briefly describe this music piece.",
        "What does this music sound like? Describe it.",
        "Describe the overall feel and character of this track.",
        "Summarize this piece of music in your own words.",
        "Give an overview of what this music sounds like.",
        "Describe the style and mood of this audio clip.",
        "Write a description of this music piece.",
        "Describe the musical characteristics of this track.",
        "What are the main features of this piece of music?",
        "How would you describe this track to someone?",
        "Provide a description of this musical piece.",
        "Describe the sound and feel of this music.",
    ],
    ("sections", "zh"): [
        "请按这段音乐本身的组织顺序，概括它涉及到的几个方面，并据此描述这段音乐。",
        "请按照合适的顺序，从这段音乐体现出的几个主要方面来描述它。",
        "请根据这段音乐适合的展开顺序，从几个主要维度对它进行描述。",
        "请从节奏、配器、结构和情绪等方面描述这段音乐。",
        "请从整体氛围、乐器音色、结构变化和情绪表达几个方面描述这段音乐。",
        "请分别从风格、编曲、人声表现和情感几个角度描述这段音乐。",
        "请描述这段音乐的演奏技法、乐器音色、结构变化和整体情绪。",
        "请从配器编排、动态变化和调性色彩几个方面分析这段音乐。",
    ],
    ("sections", "en"): [
        "Describe this track by covering the main musical aspects in a natural order.",
        "Write a structured description of this music piece, "
        "following an appropriate order of musical aspects.",
        "Describe this audio clip by organizing the main musical dimensions "
        "in a natural sequence.",
        "Describe this piece by discussing its rhythm, instrumentation, "
        "structure, and mood.",
        "Describe the overall atmosphere, vocal character, harmonic "
        "progression, and emotional tone of this track.",
        "Write a description covering the production style, arrangement, "
        "dynamics, and mood of this music.",
        "Describe this track's instrumentation, structural development, "
        "tonal character, and emotional quality.",
        "Discuss this piece in terms of its sound design, rhythmic feel, "
        "melodic character, and overall atmosphere.",
    ],
}

FEWSHOT_EXAMPLES = r"""
Examples (PARAGRAPH_EN — vary phrasing, do NOT repeat these exactly):
1) {"prompt":"Describe this piece of music."}
2) {"prompt":"What does this track sound like?"}
3) {"prompt":"Give an overview of this music piece."}
4) {"prompt":"Describe the overall feel and character of this track."}
5) {"prompt":"Summarize this piece of music in your own words."}
6) {"prompt":"Describe the overall character and feel of this musical piece in a detailed paragraph."}

Examples (PARAGRAPH_ZH — vary phrasing, do NOT repeat these exactly):
1) {"prompt":"请描述一下这段音乐。"}
2) {"prompt":"这段音乐听起来怎么样？请描述一下。"}
3) {"prompt":"请概括一下这段音乐的整体特征。"}
4) {"prompt":"请详细描述一下这段音乐的整体风格和听觉感受。"}
5) {"prompt":"请描述你从这段音乐中听到了什么。"}
6) {"prompt":"用你的话介绍一下这段音乐的风格和特点。"}

Examples (SECTIONS_EN — vary the aspects and their ordering):
1) {"prompt":"Describe this track by discussing its rhythmic feel, instrumentation and arrangement, structural development, vocal features if present, and overall mood."}
2) {"prompt":"Write a description of this music piece, covering the overall atmosphere, production texture, dynamic flow, tonal character, and emotional theme."}
3) {"prompt":"Describe this piece by exploring its overall atmosphere, instrumentation and arrangement, structural development, harmonic character, and the emotional quality conveyed throughout the track."}

Examples (SECTIONS_ZH — vary the aspects and their ordering):
1) {"prompt":"请按顺序描述这段音乐的节奏与律动、配器与编排、结构发展、是否有人声，以及整体情绪。"}
2) {"prompt":"请从整体氛围、制作质感、动态变化、调性色彩和情绪主题几个方面描述这段音乐。"}
3) {"prompt":"请从整体氛围、乐器音色与演奏方式、结构发展与调性变化、动态起伏，以及情绪表达几个方面描述这段音乐。"}
""".strip()

PARAGRAPH_PROMPT_TEMPLATE = r"""
You write ONE natural user instruction (a query) asking for a description of an audio clip.

Inputs:
- A target caption.
- A language: ZH or EN.
- A length class: SHORT / MEDIUM / LONG.

Goal:
Generate a user query for a PARAGRAPH-style response.

Rules:
1. Ask only for a general description of the music.
2. Do NOT enumerate dimensions.
3. Keep it natural, simple, and instruction-following.
4. The query must remain generic.
5. Do NOT copy or imply any specific facts from the target caption.
6. Do NOT mention dataset, caption, metadata, JSON, target text, analysis, timestamps, or source comparison.
7. If language is ZH, output Chinese.
8. If language is EN, output English.
9. IMPORTANT: Vary your phrasing. Use the Seed value to inspire different word choices, sentence structures, and perspectives. Do NOT always produce the same query.

Length constraints:
- SHORT: one very short sentence.
- MEDIUM: one natural short sentence.
- LONG: one natural sentence, slightly richer if needed, but still not section-like.

Output format:
- Output ONLY one-line JSON: {{"prompt":"..."}}
- No extra keys.
- No code fences.
- No explanations.

{FEWSHOTS}

Language:
{LANGUAGE_CLASS}

Length class:
{LENGTH_CLASS}

Seed:
{SEED}

Target caption (for style reference only; do NOT copy specific content):
{CAPTION}
""".strip()

SECTIONS_PROMPT_TEMPLATE = r"""
You write ONE natural user instruction (a query) asking for a description of an audio clip.

Inputs:
- A target caption.
- A language: ZH or EN.
- A length class: SHORT / MEDIUM / LONG.

Goal:
Generate a user query for a SECTIONS-style response.

Core task:
- Infer the ORDER of musical aspects from the organizational order of the target caption.
- Then convert that inferred order into a natural user request.
- The requested aspects must stay generic and high-level.

Allowed aspect types:
- overall sound
- style / genre feel
- atmosphere
- tempo / rhythm / groove
- instrumentation / arrangement / production
- structure / dynamics / progression
- vocals / lyrics if present
- tonal color
- mood / emotion / theme
- other similarly generic musical dimensions

Important constraints:
1. Do NOT force a fixed canonical order. Follow the apparent order suggested by the target caption.
2. Do NOT copy specific facts, labels, quoted phrases, artist names, languages, instruments, cultures, regions, or distinctive descriptors from the target caption.
3. Keep the query generic.
4. The query should sound like a natural user request, not a schema or checklist.
5. Do NOT mention dataset, caption, metadata, JSON, analysis process, timestamps, or source comparison.
6. If language is ZH, output Chinese.
7. If language is EN, output English.
8. IMPORTANT: Vary your phrasing and the selection/order of aspects. Use the Seed value to inspire different word choices. Do NOT always produce the same query.

Length constraints:
- SHORT: one short sentence that briefly reflects the inferred order.
- MEDIUM: one sentence or two short sentences with the inferred order of aspects.
- LONG: one or two natural sentences, richer but still concise.

Output format:
- Output ONLY one-line JSON: {{"prompt":"..."}}
- No extra keys.
- No code fences.
- No explanations.

{FEWSHOTS}

Language:
{LANGUAGE_CLASS}

Length class:
{LENGTH_CLASS}

Seed:
{SEED}

Target caption (for style/order reference only; do NOT copy specific content):
{CAPTION}
""".strip()


# ─── Utils ────────────────────────────────────────────────────────────────────

def count_lines_fast(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def normalize_style(x: str) -> str:
    x = (x or "paragraph").strip().lower()
    return "sections" if x == "sections" else "paragraph"


def normalize_language(x: str) -> str:
    x = (x or "en").strip().lower()
    return "zh" if x.startswith("zh") else "en"


def stable_length_by_style(stable_key: str, style_class: str) -> str:
    style_class = normalize_style(style_class)
    h = hashlib.md5(f"{stable_key}:{style_class}".encode("utf-8")).hexdigest()
    x = int(h[:8], 16) % 100
    if style_class == "sections":
        return "SHORT" if x < 15 else ("MEDIUM" if x < 70 else "LONG")
    else:
        return "SHORT" if x < 60 else ("MEDIUM" if x < 90 else "LONG")


def make_seed(stable_key: str) -> str:
    h = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()
    return h[:8]


def build_prompt_for_generator(caption: str, length_class: str, seed: str,
                               style_class: str, language: str) -> str:
    cap = (caption or "").strip() or "(no available description)"
    style_class = normalize_style(style_class)
    language = normalize_language(language)
    language_name = "ZH" if language == "zh" else "EN"
    template = SECTIONS_PROMPT_TEMPLATE if style_class == "sections" \
        else PARAGRAPH_PROMPT_TEMPLATE
    return template.format(
        FEWSHOTS=FEWSHOT_EXAMPLES, LANGUAGE_CLASS=language_name,
        LENGTH_CLASS=length_class, SEED=seed, CAPTION=cap,
    )


def is_jsonl_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(".jsonl")


def looks_like_jsonl_path(path: str) -> bool:
    return path.endswith(".jsonl")


def resolve_input_files(input_path: str):
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.jsonl")))
        return [(os.path.basename(f), f) for f in files]
    if is_jsonl_file(input_path):
        return [(os.path.basename(input_path), input_path)]
    raise ValueError(f"input_path must be a directory or .jsonl: {input_path}")


def resolve_output_mode(input_files, output_path: str):
    if not input_files:
        raise ValueError("No input files resolved.")

    if os.path.isdir(output_path):
        mapping = {n: os.path.join(output_path, n) for n, _ in input_files}
        return "directory", mapping, os.path.join(output_path, "failed_records.log")

    if looks_like_jsonl_path(output_path):
        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        mapping = {n: output_path for n, _ in input_files}
        return "single_file", mapping, output_path + ".failed.log"

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        mapping = {n: os.path.join(output_path, n) for n, _ in input_files}
        return "directory", mapping, os.path.join(output_path, "failed_records.log")

    raise ValueError(f"Invalid output_path: {output_path}")


def read_processed_counts(output_mode: str, file_to_output_path: dict,
                          input_files):
    processed = {}
    if output_mode == "single_file":
        for name, _ in input_files:
            processed[name] = 0
        return processed

    for name, _ in input_files:
        out_path = file_to_output_path[name]
        if not os.path.exists(out_path):
            processed[name] = 0
            continue
        c = 0
        with open(out_path, "rb") as f:
            for _ in f:
                c += 1
        processed[name] = c
    return processed


# ─── API ──────────────────────────────────────────────────────────────────────

async def call_llm(session: aiohttp.ClientSession, api_url: str,
                   api_key: str, model_name: str, user_prompt: str,
                   temperature: float = 0.8) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system",
             "content": "You generate natural user queries for music audio captioning. "
                        "Follow the requested style and output format exactly."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    async with session.post(api_url, headers=headers, json=payload) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
        data = json.loads(text)
        return data["choices"][0]["message"]["content"].strip()


def parse_prompt_json(s: str) -> str:
    s = s.strip()
    try:
        obj = json.loads(s)
    except Exception:
        l = s.find("{")
        r = s.rfind("}")
        if l == -1 or r == -1 or r <= l:
            raise
        obj = json.loads(s[l:r + 1])
    p = obj.get("prompt", "")
    if not isinstance(p, str) or not p.strip():
        raise ValueError("missing/empty prompt field")
    return p.strip()


def choose_fallback(style_class: str, language: str,
                    rng: random.Random) -> str:
    style_class = normalize_style(style_class)
    language = normalize_language(language)
    pool = FALLBACK_PROMPTS[(style_class, language)]
    return rng.choice(pool)


# ─── Worker ───────────────────────────────────────────────────────────────────

async def worker(
    name: str, session: aiohttp.ClientSession, queue: asyncio.Queue,
    sem: asyncio.Semaphore, out_handles: dict, failed_fh, pbar: tqdm,
    retries: int, api_url: str, api_key: str, model_name: str,
    file_to_output_path: dict,
):
    while True:
        item = await queue.get()
        try:
            if item is None:
                break

            file_name, idx, record = item
            audio_path = record.get("audio_path")
            caption = record.get("caption")
            caption_style = normalize_style(record.get("caption_style", "paragraph"))
            caption_language = normalize_language(
                record.get("caption_language", "en"))

            if not audio_path or caption is None:
                failed_fh.write(json.dumps({
                    "file": file_name, "idx": idx,
                    "audio_path": str(audio_path),
                    "error": "missing audio_path or caption",
                }, ensure_ascii=False) + "\n")
                continue

            stable_key = f"{file_name}:{idx}:{audio_path}"
            length_class = stable_length_by_style(stable_key, caption_style)
            seed = make_seed(stable_key)

            gen_prompt = build_prompt_for_generator(
                caption=str(caption), length_class=length_class,
                seed=seed, style_class=caption_style, language=caption_language,
            )

            llm_out = ""
            last_err = ""
            async with sem:
                for attempt in range(retries):
                    try:
                        llm_out = await call_llm(
                            session, api_url, api_key, model_name,
                            gen_prompt, temperature=0.9,
                        )
                        break
                    except Exception as e:
                        last_err = str(e)
                        if attempt < retries - 1:
                            await asyncio.sleep(1.5)

            user_query = ""
            if llm_out:
                try:
                    user_query = parse_prompt_json(llm_out)
                except Exception as e:
                    last_err = (f"parse prompt json failed: {e} | "
                                f"raw={llm_out[:300]!r}")

            if not user_query:
                rng = random.Random(int(seed, 16))
                user_query = choose_fallback(caption_style, caption_language, rng)
                if last_err:
                    failed_fh.write(json.dumps({
                        "file": file_name, "idx": idx,
                        "audio_path": str(audio_path),
                        "error": last_err, "fallback_used": True,
                    }, ensure_ascii=False) + "\n")

            out_obj = {
                "messages": [
                    {"role": "user", "content": "<audio>" + user_query},
                    {"role": "assistant", "content": str(caption)},
                ],
                "audios": [str(audio_path)],
                "meta": {
                    "source_file": file_name, "source_idx": idx,
                    "caption_style": caption_style,
                    "caption_language": caption_language,
                    "query_length_class": length_class,
                    "query_seed": seed,
                },
            }

            out_path = file_to_output_path[file_name]
            fh = out_handles.get(out_path)
            if fh is None:
                fh = open(out_path, "a", encoding="utf-8")
                out_handles[out_path] = fh
            fh.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

        except Exception as e:
            try:
                file_name, idx, record = item if item else ("<unk>", -1, {})
                audio_path = record.get("audio_path", "") \
                    if isinstance(record, dict) else ""
            except Exception:
                file_name, idx, audio_path = "<unk>", -1, ""
            failed_fh.write(json.dumps({
                "file": file_name, "idx": idx,
                "audio_path": str(audio_path),
                "error": f"worker fatal: {e}",
            }, ensure_ascii=False) + "\n")
        finally:
            queue.task_done()
            if item is not None:
                pbar.update(1)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main_async(
    input_path: str, output_path: str, api_url: str, api_key: str,
    model_name: str, concurrency: int, retries: int, timeout_sec: int,
):
    input_files = resolve_input_files(input_path)
    if not input_files:
        print("[INFO] No jsonl files found.", file=sys.stderr)
        return

    output_mode, file_to_output_path, failed_log_path = \
        resolve_output_mode(input_files, output_path)

    input_files = [
        (name, path) for name, path in input_files
        if os.path.abspath(path) not in
           {os.path.abspath(p) for p in set(file_to_output_path.values())}
    ]
    if not input_files:
        print("[INFO] No valid input files after excluding output.", file=sys.stderr)
        return

    processed_count = read_processed_counts(
        output_mode, file_to_output_path, input_files)
    already = sum(processed_count.values())
    total_lines = sum(count_lines_fast(fp) for _, fp in input_files)

    print(f"[INFO] {len(input_files)} file(s), {total_lines} lines, "
          f"already={already}", file=sys.stderr)

    timeout = ClientTimeout(total=timeout_sec)
    connector = aiohttp.TCPConnector(limit=concurrency * 2, ttl_dns_cache=300)

    os.makedirs(os.path.dirname(os.path.abspath(failed_log_path)) or ".",
                exist_ok=True)

    if output_mode == "single_file":
        open(output_path, "w", encoding="utf-8").close()

    out_handles: dict = {}
    failed_fh = open(failed_log_path, "a", encoding="utf-8")
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=total_lines, initial=min(already, total_lines),
                desc="Query gen", unit="line")

    try:
        async with aiohttp.ClientSession(timeout=timeout,
                                         connector=connector) as session:
            worker_tasks = [
                asyncio.create_task(worker(
                    f"w{i}", session, queue, sem, out_handles, failed_fh,
                    pbar, retries, api_url, api_key, model_name,
                    file_to_output_path,
                ))
                for i in range(concurrency)
            ]

            for fp_name, fp in input_files:
                skip = processed_count.get(fp_name, 0)
                with open(fp, "r", encoding="utf-8") as f:
                    for _ in range(skip):
                        if not f.readline():
                            break
                    idx = skip
                    for line in f:
                        idx += 1
                        line = line.strip()
                        if not line:
                            pbar.update(1)
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            pbar.update(1)
                            continue
                        await queue.put((fp_name, idx, record))

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
        pbar.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="根据 caption 生成训练用 query（对话格式数据）")
    p.add_argument("input_path", help="输入 jsonl 文件或目录")
    p.add_argument("output_path", help="输出 jsonl 文件或目录")
    p.add_argument("--api_url", type=str, required=True,
                    help="Chat completions API URL")
    p.add_argument("--model", type=str, required=True, help="LLM 模型名称")
    p.add_argument("--api_key_env", default="INF_API_KEY",
                    help="API Key 环境变量名")
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout_sec", type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"[FATAL] 环境变量 {args.api_key_env} 未设置", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main_async(
        input_path=args.input_path, output_path=args.output_path,
        api_url=args.api_url, api_key=api_key, model_name=args.model,
        concurrency=args.concurrency, retries=args.retries,
        timeout_sec=args.timeout_sec,
    ))


if __name__ == "__main__":
    main()
