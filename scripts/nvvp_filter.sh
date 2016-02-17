#!/usr/bin/sh
#
# Taken from: http://uob-hpc.github.io/2015/05/27/nvvp-import-opencl/

filter="
s/OPENCL_/CUDA_/g
s/ndrange/grid/g
s/workitem/thread/g
s/workgroupsize/threadblocksize/g
s/stapmemperworkgroup/stasmemperblock/g
"

sed -e "$filter" "$1"
