#!/usr/bin/env python3
"""
合并分片处理后的 results.jsonl 文件。

MusicToolsPipeline 在 group_by_segment=True 模式下，每个 segment 会输出到
独立子目录（如 segment_00/results.jsonl, segment_01/results.jsonl …）。
本脚本将这些分散的 results.jsonl 合并为一个完整文件。

用法:
    python merge_sharded_results.py output_dir/ merged_results.jsonl
    python merge_sharded_results.py output_dir/ merged_results.jsonl --results-filename results.jsonl
"""

import argparse
import os
import sys


def merge_sharded_results(
    sharded_dir: str,
    output_path: str,
    results_filename: str = "results.jsonl",
) -> int:
    subdirs = sorted(
        d for d in os.listdir(sharded_dir)
        if os.path.isdir(os.path.join(sharded_dir, d))
    )

    if not subdirs:
        print(f"警告: {sharded_dir} 下无子目录", file=sys.stderr)
        return 0

    result_files = []
    for d in subdirs:
        p = os.path.join(sharded_dir, d, results_filename)
        if os.path.isfile(p):
            result_files.append(p)
        else:
            print(f"  跳过 {d}/（缺少 {results_filename}）")

    if not result_files:
        print(f"错误: 未找到任何 {results_filename} 文件", file=sys.stderr)
        return 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_lines = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for rf in result_files:
            with open(rf, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line if line.endswith("\n") else line + "\n")
                        total_lines += 1
            print(f"  合并: {os.path.relpath(rf, sharded_dir)}")

    print(f"合并完成: {len(result_files)} 个文件, {total_lines} 条记录 → {output_path}")
    return total_lines


def main():
    parser = argparse.ArgumentParser(
        description="合并 MusicToolsPipeline 分片输出的 results.jsonl"
    )
    parser.add_argument("sharded_dir", help="分片输出的根目录（包含 segment_XX/ 子目录）")
    parser.add_argument("output", help="合并后的输出文件路径")
    parser.add_argument(
        "--results-filename", default="results.jsonl",
        help="子目录内结果文件名（默认 results.jsonl）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.sharded_dir):
        print(f"错误: 目录不存在: {args.sharded_dir}", file=sys.stderr)
        sys.exit(1)

    count = merge_sharded_results(args.sharded_dir, args.output, args.results_filename)
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
