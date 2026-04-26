#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3a — 调用 Qwen3-Omni ASR 服务，对裁剪后的音频片段进行歌词识别。

采用原子重写策略：失败重跑后会覆盖旧的失败记录。
输出字段：qwen3-omni-asr-result
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
from tqdm import tqdm

PROMPT_POOL = [
    (
        "You are an accurate ASR system for singing vocals.\n"
        "Task: Transcribe intelligible vocals into text/lyrics.\n"
        "Output plain text only.\n"
        "Rules:\n"
        "1) Preserve line breaks and repetitions.\n"
        "2) Do NOT invent words. If unsure/unintelligible, use [inaudible].\n"
        "3) If there are NO intelligible vocals/lyrics in this audio, "
        "output exactly: NO_LYRICS_DETECTED\n"
    )
]

_LANG_TAG_RE = re.compile(r"^<\|[a-z_]+\|>\s*")


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https", "file")
    except Exception:
        return False


def normalize_audio_path(audio_path: str) -> str:
    if is_url(audio_path):
        return audio_path
    p = Path(audio_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def to_audio_url(audio_path: str) -> str:
    if is_url(audio_path):
        return audio_path
    p = Path(audio_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    return "file:///" + str(p).lstrip("/")


def build_messages(prompt_text: str, audio_url: str,
                    *, audio_only: bool = False):
    if audio_only:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                ],
            }
        ]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ],
        }
    ]


def strip_lang_tag(text: str) -> str:
    """Strip Qwen3-ASR language prefix like ``<|zh|>`` or ``<|en|>``."""
    return _LANG_TAG_RE.sub("", text)


def collect_jsonl_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for x in inputs:
        p = Path(x)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file() and p.suffix == ".jsonl":
            paths.append(p)
    return sorted(set(paths))


def count_jsonl_lines(paths: List[Path]) -> int:
    total = 0
    for p in paths:
        with p.open("rb") as f:
            total += sum(1 for _ in f)
    return total


def get_resume_key(obj: Dict[str, Any], resume_key: str,
                   audio_field: str) -> Optional[str]:
    if resume_key:
        v = obj.get(resume_key)
        if isinstance(v, (str, int)):
            return str(v)
    v2 = obj.get(audio_field)
    if isinstance(v2, str) and v2.strip():
        return normalize_audio_path(v2.strip())
    return None


def is_success(obj: Dict[str, Any], output_field: str, error_field: str) -> bool:
    out = obj.get(output_field)
    err = obj.get(error_field)
    return isinstance(out, str) and out.strip() and not (isinstance(err, str) and err.strip())


def normalize_record_audio_path(obj: Dict[str, Any],
                                audio_field: str) -> Dict[str, Any]:
    obj2 = dict(obj)
    audio_path = obj2.get(audio_field)
    if isinstance(audio_path, str) and audio_path.strip():
        obj2[audio_field] = normalize_audio_path(audio_path.strip())
    return obj2


def load_existing_records_map(
    out_path: Path, resume_key: str, audio_field: str,
) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not out_path.exists():
        return records
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj = normalize_record_audio_path(obj, audio_field)
            k = get_resume_key(obj, resume_key, audio_field)
            if k is not None:
                records[k] = obj
    return records


async def post_chat(
    session: aiohttp.ClientSession, base_url: str, model: str,
    messages: List[Dict[str, Any]], max_tokens: int,
    temperature: float, timeout_s: int,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=payload, timeout=timeout) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {text[:800]}")
        data = json.loads(text)

    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return (part.get("text") or "").strip()
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


async def infer_one(
    session: aiohttp.ClientSession, servers: List[str],
    server_rr_lock: asyncio.Lock, rr_state: Dict[str, int],
    model: str, obj: Dict[str, Any], audio_field: str,
    output_field: str, error_field: str, max_tokens: int,
    temperature: float, timeout_s: int, retries: int,
    retry_base_sleep: float,
    *, audio_only: bool = False, do_strip_lang_tag: bool = False,
) -> Dict[str, Any]:
    use_prompt = random.choice(PROMPT_POOL)
    obj = normalize_record_audio_path(obj, audio_field)

    audio_path = obj.get(audio_field)
    if not isinstance(audio_path, str) or not audio_path.strip():
        obj[output_field] = ""
        obj[error_field] = f"missing_{audio_field}"
        return obj

    audio_url = to_audio_url(audio_path.strip())
    messages = build_messages(use_prompt, audio_url, audio_only=audio_only)

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        async with server_rr_lock:
            idx = rr_state["i"]
            rr_state["i"] = (rr_state["i"] + 1) % len(servers)
        base_url = servers[idx]

        try:
            caption = await post_chat(
                session=session, base_url=base_url, model=model,
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, timeout_s=timeout_s,
            )
            if do_strip_lang_tag:
                caption = strip_lang_tag(caption)
            obj[output_field] = caption
            obj.pop(error_field, None)
            return obj
        except Exception as e:
            last_err = e
            sleep_s = min((2 ** attempt) * retry_base_sleep + random.random() * 0.2, 8.0)
            await asyncio.sleep(sleep_s)

    obj[output_field] = ""
    obj[error_field] = f"{type(last_err).__name__}: {last_err}"
    return obj


def atomic_rewrite_jsonl(out_path: Path, records_map: Dict[str, Dict[str, Any]]):
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for obj in records_map.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)


async def process_one_file(args, session: aiohttp.ClientSession, pbar: tqdm):
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_records = load_existing_records_map(
        out_path, args.resume_key, args.audio_field)

    done_keys = {
        k for k, v in existing_records.items()
        if is_success(v, args.output_field, args.error_field)
    }

    sem = args.sem
    server_rr_lock = args.server_rr_lock
    rr_state = args.rr_state

    update_lock = asyncio.Lock()
    flush_lock = asyncio.Lock()
    dirty_count = 0

    async def flush_snapshot(snapshot: Dict[str, Dict[str, Any]]):
        async with flush_lock:
            await asyncio.to_thread(atomic_rewrite_jsonl, out_path, snapshot)

    async def update_record(obj: Dict[str, Any]):
        nonlocal dirty_count
        obj = normalize_record_audio_path(obj, args.audio_field)
        k = get_resume_key(obj, args.resume_key, args.audio_field)
        snapshot = None

        async with update_lock:
            if k is not None:
                existing_records[k] = obj
                if is_success(obj, args.output_field, args.error_field):
                    done_keys.add(k)
                else:
                    done_keys.discard(k)
                dirty_count += 1
                if dirty_count >= args.flush_every:
                    snapshot = dict(existing_records)
                    dirty_count = 0

        if snapshot is not None:
            await flush_snapshot(snapshot)

    async def run_one(raw_line: str):
        async with sem:
            line = raw_line.strip()
            if not line:
                pbar.update(1)
                return
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] json_error in {in_path.name}: {e}", file=sys.stderr)
                pbar.update(1)
                return

            obj = normalize_record_audio_path(obj, args.audio_field)
            k = get_resume_key(obj, args.resume_key, args.audio_field)

            if k is not None and k in done_keys:
                pbar.update(1)
                return

            if args.skip_existing and is_success(obj, args.output_field, args.error_field):
                await update_record(obj)
                pbar.update(1)
                return

            out_obj = await infer_one(
                session=session, servers=args.servers,
                server_rr_lock=server_rr_lock, rr_state=rr_state,
                model=args.model, obj=obj, audio_field=args.audio_field,
                output_field=args.output_field, error_field=args.error_field,
                max_tokens=args.max_tokens, temperature=args.temperature,
                timeout_s=args.timeout, retries=args.retries,
                retry_base_sleep=args.retry_base_sleep,
                audio_only=args.audio_only,
                do_strip_lang_tag=args.strip_lang_tag,
            )
            await update_record(out_obj)
            pbar.update(1)

    tasks: Set[asyncio.Task] = set()
    with in_path.open("r", encoding="utf-8") as fin:
        for raw_line in fin:
            t = asyncio.create_task(run_one(raw_line))
            tasks.add(t)
            if len(tasks) >= args.task_buffer:
                done_t, pending = await asyncio.wait(tasks,
                                                     return_when=asyncio.FIRST_COMPLETED)
                for x in done_t:
                    await x
                tasks = pending

    if tasks:
        await asyncio.gather(*tasks)

    async with update_lock:
        final_snapshot = dict(existing_records)
    await flush_snapshot(final_snapshot)


async def process_all_async(args):
    jsonl_paths = collect_jsonl_paths(args.inputs)
    if not jsonl_paths:
        print("No jsonl files found.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_lines = count_jsonl_lines(jsonl_paths)
    pbar = tqdm(total=total_lines, desc="ASR inference", unit="item",
                dynamic_ncols=True)

    connector = aiohttp.TCPConnector(
        limit=max(args.concurrency * 2, 64),
        limit_per_host=max(args.concurrency * 2, 64),
        ttl_dns_cache=300,
    )

    api_key = os.environ.get(args.api_key_env, "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        args.sem = asyncio.Semaphore(args.concurrency)
        args.server_rr_lock = asyncio.Lock()
        args.rr_state = {"i": 0}

        for in_path in jsonl_paths:
            out_path = out_dir / f"{in_path.stem}.asr.jsonl"
            args.in_path = str(in_path)
            args.out_path = str(out_path)
            await process_one_file(args, session, pbar)
            print(f"[file done] {in_path.name}", file=sys.stderr)

    pbar.close()
    print("[done]", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="调用 Qwen3-Omni ASR 服务对裁剪音频进行歌词识别")
    ap.add_argument("--inputs", "-i", nargs="+", required=True,
                    help="输入 jsonl 文件或目录")
    ap.add_argument("--out_dir", "-o", required=True, help="输出目录")
    ap.add_argument("--servers", nargs="+", required=True,
                    help="ASR 服务 Base URL（支持多个做轮询）")
    ap.add_argument("--api_key_env", default="INF_API_KEY")
    ap.add_argument("--model", required=True, help="ASR 模型名称或路径")
    ap.add_argument("--audio_field", default="audio_path")
    ap.add_argument("--output_field", default="qwen3-omni-asr-result")
    ap.add_argument("--error_field", default="error")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--retry_base_sleep", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=2048)
    ap.add_argument("--task_buffer", type=int, default=10000)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--resume_key", type=str, default="")
    ap.add_argument("--flush_every", type=int, default=1000,
                    help="每处理多少条记录原子重写一次")
    ap.add_argument("--audio_only", action="store_true",
                    help="仅发送音频 (不含 text prompt), 用于 Qwen3-ASR 等纯 ASR 模型")
    ap.add_argument("--strip_lang_tag", action="store_true",
                    help="去除 Qwen3-ASR 输出的 <|lang|> 前缀")
    args = ap.parse_args()
    rc = asyncio.run(process_all_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
