#!/bin/bash -e

# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

source /arm/tools/setup/init/bash
module load swdev python/python/3.8.5
export LD_LIBRARY_PATH="/arm/tools/gnu/gcc/7.3.0/rhe7-x86_64/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="`realpath ../..`:${PYTHONPATH}"

python3 -m pylint AIPUBuilder --rcfile=AIPUBuilder/Optimizer/scripts/lint/pylintrc
