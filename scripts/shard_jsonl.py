#!/usr/bin/env python3
"""
将 JSONL 文件分片为多个 segment_XX.jsonl 文件，用于 MusicToolsPipeline 目录模式。

用法:
    python shard_jsonl.py input.jsonl output_dir/ --shard-size 25000
"""

import argparse
import os
import sys


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def shard_jsonl(input_path: str, output_dir: str, shard_size: int) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)

    total = count_lines(input_path)
    if total == 0:
        print("输入文件为空，无需分片")
        return []

    num_shards = (total + shard_size - 1) // shard_size
    print(f"总行数: {total}, 分片大小: {shard_size}, 预计分片数: {num_shards}")

    created: list[str] = []
    shard_idx = 0
    line_count = 0
    out_f = None

    with open(input_path, "r", encoding="utf-8") as in_f:
        for line in in_f:
            if line_count % shard_size == 0:
                if out_f is not None:
                    out_f.close()
                shard_name = f"segment_{shard_idx:02d}.jsonl"
                shard_path = os.path.join(output_dir, shard_name)
                out_f = open(shard_path, "w", encoding="utf-8")
                created.append(shard_path)
                shard_idx += 1
            out_f.write(line)
            line_count += 1

    if out_f is not None:
        out_f.close()

    for i, p in enumerate(created):
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            n = sum(1 for _ in f)
        print(f"  {os.path.basename(p)}: {n} 条 ({size / 1024 / 1024:.1f} MB)")

    print(f"分片完成: {len(created)} 个文件 → {output_dir}")
    return created


def main():
    parser = argparse.ArgumentParser(description="将 JSONL 文件分片为 segment_XX.jsonl")
    parser.add_argument("input", help="输入 JSONL 文件路径")
    parser.add_argument("output_dir", help="输出目录（存放 segment_XX.jsonl）")
    parser.add_argument(
        "--shard-size", type=int, default=25000,
        help="每个分片的最大行数（默认 25000）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"错误: 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.shard_size <= 0:
        print(f"错误: --shard-size 必须为正整数", file=sys.stderr)
        sys.exit(1)

    shard_jsonl(args.input, args.output_dir, args.shard_size)


if __name__ == "__main__":
    main()
