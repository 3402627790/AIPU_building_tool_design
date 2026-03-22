#!/bin/bash -e
#
# after submitted to lsf host, some environment value changed
# setup environment for every bsub jobs here
# 
source /arm/tools/setup/init/bash
module load swdev python/python/3.10.12

export LD_LIBRARY_PATH=/arm/tools/gnu/gcc/9.3.0/rhe7-x86_64/lib64:$LD_LIBRARY_PATH

$@