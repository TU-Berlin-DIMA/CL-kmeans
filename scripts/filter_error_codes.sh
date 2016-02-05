#!/bin/sh

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
# 
# 
# Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>

# Filter cl.h file for OpenCL error codes
# Print error codes as C/C++ switch/case
#
# Expected input: cl.h

cl_header_file="$1"

if [ -n $cl_header_file ]
then
    echo "Missing input"
    echo "Expected \"cl.h\" file"
    exit -1
fi

# https://nixtip.wordpress.com/2010/10/12/print-lines-between-two-patterns-the-awk-way/

echo "switch(error_code) {"

awk '
/\/\*\s+Error/{ flag = 1; next }
/\/\*/{ flag = 0 }
flag && /#define/ { print "case " $2 ":\n    s=\"" $2 "\";\n    break;" }
' $cl_header_file \
    | sed 's/^/    /'

echo "}"
