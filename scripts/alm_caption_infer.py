#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1a — 调用 Audio-Language Model (ALM) 服务，为每首歌生成 Base Caption。

兼容任意 OpenAI-compatible 的多模态 chat 接口，典型选项包括：
  - Qwen3-Omni 系列（Qwen3-Omni-30B-A3B-Instruct 等）
  - MusicFlamingo / Audio-Flamingo-3 等专用音乐 / 音频 ALM
  - 其他支持 `audio_url` 的 vLLM / sglang / tgi 服务

依赖：上述 ALM 服务已启动（见 README 部署说明）。
输出字段默认：ALM_Caption（可通过 --output_field 覆盖）。
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
from tqdm import tqdm

PROMPT_POOL = [
    "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
    "Write a rich caption that blends the technical details (genre, BPM, key, chords, mix) with how the song feels emotionally and dynamically as it unfolds.",
    "Create a descriptive music caption that combines technical aspects (style, tempo feel, harmony, sound design) with a narrative of the song's emotional arc from beginning to end.",
    "Analyze the track and write a cohesive paragraph describing its musical style, tempo and harmonic character, key instruments and mix, and the mood or atmosphere it conveys as it progresses.",
    "Write a musically informed caption that weaves together genre, rhythmic intensity, harmonic feel, instrumentation, production texture, and the emotional journey of the piece.",
]


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https", "file")
    except Exception:
        return False


def to_audio_url(audio_path: str) -> str:
    if is_url(audio_path):
        return audio_path
    p = Path(audio_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return "file:///" + str(p).lstrip("/")


def build_messages(prompt_text: str, audio_url: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ],
        }
    ]


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
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def get_resume_key(obj: Dict[str, Any], resume_key: str,
                   audio_field: str) -> Optional[str]:
    if resume_key:
        v = obj.get(resume_key)
        if isinstance(v, (str, int)):
            return str(v)
    v2 = obj.get(audio_field)
    if isinstance(v2, str) and v2.strip():
        return v2.strip()
    return None


def load_done_keys(out_path: Path, resume_key: str,
                   audio_field: str) -> Set[str]:
    done: Set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = get_resume_key(obj, resume_key, audio_field)
            if k is not None:
                done.add(k)
    return done


async def post_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
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
    session: aiohttp.ClientSession,
    servers: List[str],
    server_rr_lock: asyncio.Lock,
    rr_state: Dict[str, int],
    model: str,
    obj: Dict[str, Any],
    audio_field: str,
    output_field: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    retries: int,
    retry_base_sleep: float,
) -> Dict[str, Any]:
    use_prompt = random.choice(PROMPT_POOL)

    audio_path = obj.get(audio_field)
    if not isinstance(audio_path, str) or not audio_path.strip():
        obj[output_field] = ""
        obj["_prompt"] = use_prompt
        obj["_error"] = f"missing_{audio_field}"
        return obj

    audio_url = to_audio_url(audio_path.strip())
    messages = build_messages(use_prompt, audio_url)

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
            obj[output_field] = caption
            obj["_alm_prompt"] = use_prompt
            return obj
        except Exception as e:
            last_err = e
            sleep_s = min((2 ** attempt) * retry_base_sleep + random.random() * 0.2, 8.0)
            await asyncio.sleep(sleep_s)

    obj[output_field] = ""
    obj["_alm_prompt"] = use_prompt
    obj["_error"] = f"infer_error: {type(last_err).__name__}: {last_err}"
    return obj


async def process_one_file(args, session: aiohttp.ClientSession, pbar: tqdm):
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = load_done_keys(out_path, args.resume_key, args.audio_field)
    sem = args.sem
    server_rr_lock = args.server_rr_lock
    rr_state = args.rr_state

    write_lock = asyncio.Lock()
    fout = out_path.open("a", encoding="utf-8")

    async def write_line(obj: Dict[str, Any]):
        async with write_lock:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()

    async def run_one(raw_line: str):
        async with sem:
            line = raw_line.strip()
            if not line:
                pbar.update(1)
                return
            try:
                obj = json.loads(line)
            except Exception as e:
                await write_line({"_raw": line, "_error": f"json_error: {e}"})
                pbar.update(1)
                return

            k = get_resume_key(obj, args.resume_key, args.audio_field)
            if k is not None and k in done_keys:
                pbar.update(1)
                return

            if args.skip_existing and isinstance(obj.get(args.output_field), str) \
                    and obj[args.output_field].strip():
                if k is not None:
                    done_keys.add(k)
                await write_line(obj)
                pbar.update(1)
                return

            out_obj = await infer_one(
                session=session, servers=args.servers,
                server_rr_lock=server_rr_lock, rr_state=rr_state,
                model=args.model, obj=obj, audio_field=args.audio_field,
                output_field=args.output_field, max_tokens=args.max_tokens,
                temperature=args.temperature, timeout_s=args.timeout,
                retries=args.retries, retry_base_sleep=args.retry_base_sleep,
            )
            k2 = get_resume_key(out_obj, args.resume_key, args.audio_field)
            if k2 is not None:
                done_keys.add(k2)
            await write_line(out_obj)
            pbar.update(1)

    tasks: Set[asyncio.Task] = set()
    try:
        with in_path.open("r", encoding="utf-8") as fin:
            for raw_line in fin:
                t = asyncio.create_task(run_one(raw_line))
                tasks.add(t)
                if len(tasks) >= args.task_buffer:
                    _done, tasks = await asyncio.wait(tasks,
                                                      return_when=asyncio.FIRST_COMPLETED)
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        fout.close()


async def process_all_async(args):
    jsonl_paths = collect_jsonl_paths(args.inputs)
    if not jsonl_paths:
        print("No jsonl files found.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_lines = count_jsonl_lines(jsonl_paths)
    pbar = tqdm(total=total_lines, desc="ALM caption inference", unit="item",
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
            out_path = out_dir / f"{in_path.stem}.alm.jsonl"
            args.in_path = str(in_path)
            args.out_path = str(out_path)
            await process_one_file(args, session, pbar)
            print(f"[file done] {in_path.name}", file=sys.stderr)

    pbar.close()
    print("[done]", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="调用通用 Audio-Language Model (Qwen3-Omni 系列 / "
                    "MusicFlamingo 等) 为音频生成 Base Caption。",
    )
    ap.add_argument("--inputs", "-i", nargs="+", required=True,
                    help="输入 jsonl 文件或包含 jsonl 的目录")
    ap.add_argument("--out_dir", "-o", required=True, help="输出目录")
    ap.add_argument("--servers", nargs="+", required=True,
                    help="ALM 服务 Base URL（支持多个做轮询，OpenAI 兼容接口）")
    ap.add_argument("--api_key_env", default="INF_API_KEY",
                    help="API Key 环境变量名")
    ap.add_argument("--model", required=True,
                    help="ALM 模型名，例如 Qwen3-Omni-30B-A3B-Instruct / "
                         "audio-flamingo-3 等")
    ap.add_argument("--audio_field", default="audio_path")
    ap.add_argument("--output_field", default="ALM_Caption")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry_base_sleep", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=2048,
                    help="最大并发请求数")
    ap.add_argument("--task_buffer", type=int, default=2048)
    ap.add_argument("--skip_existing", action="store_true",
                    help="输入已有 output_field 时直接保留")
    ap.add_argument("--resume_key", type=str, default="",
                    help="用作断点续跑唯一 key 的字段名")
    args = ap.parse_args()
    rc = asyncio.run(process_all_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
