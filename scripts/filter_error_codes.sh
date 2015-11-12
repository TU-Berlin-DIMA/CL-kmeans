#!/bin/sh

cl_header_file="$1"

awk ' /#define\s+\w+\s+-/ {print "case " $2 ":\n    s=\"" \
    $2 "\";\n    break;"}' $cl_header_file
