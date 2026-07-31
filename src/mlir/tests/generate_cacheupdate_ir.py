#!/usr/bin/env python3
"""Generate a minimal CacheUpdate IR using the Python ops API."""

import argparse
import os

from AIPUBuilder import ops
from AIPUBuilder.core import Dtype, Graph


def build_cacheupdate_graph(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ir_txt = os.path.join(output_dir, "cacheupdate.def")
    ir_bin = os.path.join(output_dir, "cacheupdate.bin")

    with Graph("cacheupdate_ops_api") as graph:
        cache = ops.input([1, 1, 32, 128], Dtype.FP32, name="cache")
        position_id = ops.input([1, 1], Dtype.INT32, name="position_id")
        value = ops.input([1, 4, 1, 32], Dtype.FP32, name="value")

        current_cache, total_cache = ops.cacheupdate(
            cache, position_id, value, block_index=0, axis=2
        )
        current_cache.name = "current_cache"
        total_cache.name = "total_cache"

    graph.serialize(ir_txt, ir_bin)
    return ir_txt, ir_bin


def main():
    parser = argparse.ArgumentParser(
        description="Generate CacheUpdate Compass IR using AIPUBuilder.ops."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory to write cacheupdate.def and cacheupdate.bin.",
    )
    args = parser.parse_args()

    ir_txt, ir_bin = build_cacheupdate_graph(args.output_dir)
    print(f"generated: {ir_txt}")
    print(f"generated: {ir_bin}")


if __name__ == "__main__":
    main()
