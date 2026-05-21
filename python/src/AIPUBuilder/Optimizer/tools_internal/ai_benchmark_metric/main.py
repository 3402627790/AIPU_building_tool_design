# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import argparse
import os

from config.config import AIBConfig
from aibmaster import AIBMaster

abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.environ['AIPUBUILDER_LOG'] = '2'


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ai_benchmark models test",
        description="",
    )
    parser.add_argument("-m", "--model_name", action="store", help="ai benchmark model name")
    parser.add_argument("-o", "--output_dir", default=os.path.join(dirname,
                        'output_dir'), help="ai benchmark model name")
    parser.add_argument("-b", "--backend", default='executor', help="ai benchmark model name")
    parser.add_argument("-d", "--dataset", default='NumpyDataset', help="ai benchmark model name")
    parser.add_argument("--only_forward", action='store_true', help="ai benchmark model name")
    parser.add_argument("--sim_x3_itp", default='none', type=str, help="ai benchmark model name")
    parser.add_argument("--sim_x3_itp_fp_bits", default=32, type=int, help="ai benchmark model name")

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.model_name)
    aib_config = AIBConfig(args.model_name,
                           args.backend,
                           output_dir,
                           args.dataset,
                           sim_x3_itp=args.sim_x3_itp.lower(),
                           sim_x3_itp_fp_bits=args.sim_x3_itp_fp_bits,
                           only_forward=args.only_forward)
    aib_master = AIBMaster(aib_config)
    aib_master()


if __name__ == '__main__':
    main()
