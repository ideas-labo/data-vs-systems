#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
遍历 data/ 下所有系统文件夹，统计 measurements.csv 的行数（配置数量），
并在控制台以表格形式输出。

用法：
    python analyze_measurements.py          # 使用默认的 ./data
    python analyze_measurements.py --data ./data
"""

import csv
import argparse
from pathlib import Path


def count_rows(measurements_path: Path) -> int:
    """统计 measurements.csv 的配置数量（不含表头）。"""
    with measurements_path.open("r", newline="", encoding="utf-8") as f:
        # 直接数行：总行数 - 1（表头）
        # 如果文件可能有空行，可以稍微严一点过滤掉空行
        reader = csv.reader(f)
        # 读取表头
        header = next(reader, None)
        if header is None:
            return 0
        row_count = 0
        for row in reader:
            # 跳过全空行
            if not any(cell.strip() for cell in row):
                continue
            row_count += 1
        return row_count


def analyze_data_dir(data_dir: Path):
    results = []

    # 遍历 data/ 下第一层目录（7z, Apache, brotli, ...）
    for system_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        system_name = system_dir.name
        measurements_path = system_dir / "measurements.csv"

        if measurements_path.exists():
            try:
                n_rows = count_rows(measurements_path)
                results.append((system_name, n_rows, str(measurements_path)))
            except Exception as e:
                results.append((system_name, f"ERROR: {e}", str(measurements_path)))
        else:
            # 没有 measurements.csv 的也记录一下，方便检查
            results.append((system_name, "NO measurements.csv", ""))

    # 打印结果
    print("\n=== Measurements 行数统计（配置数量）===\n")
    # 计算列宽
    name_width = max(len(r[0]) for r in results) if results else 10
    count_width = max(len(str(r[1])) for r in results) if results else 10

    header_name = "System"
    header_count = "#Configs (rows)"
    print(f"{header_name:<{name_width}}  {header_count:>{count_width}}")
    print("-" * (name_width + 2 + count_width))

    for sys_name, count, _ in results:
        print(f"{sys_name:<{name_width}}  {str(count):>{count_width}}")

    print("\n(说明：#Configs 为 measurements.csv 的非表头行数。)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="data 目录路径（默认：./data）",
    )
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    if not data_dir.exists():
        print(f"[ERROR] data 目录不存在：{data_dir}")
        return

    analyze_data_dir(data_dir)


if __name__ == "__main__":
    main()
