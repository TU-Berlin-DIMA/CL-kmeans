#!/usr/bin/env python

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
# 
# 
# Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>

import sys
import os
import struct

if len(sys.argv) != 2:
    print("Usage: file_info [FILE]")
    print("Display information about a binary format file")
    sys.exit(0)

file_name = sys.argv[1]

with open(file_name, "rb") as fin:
    header_size_bytes = 24
    raw_header = fin.read(header_size_bytes)
    (features, clusters, points) = struct.unpack('<QQQ', raw_header)
    size_bytes = os.stat(file_name).st_size - header_size_bytes

    print("Num Features: " + str(features))
    print("Num Clusters: " + str(clusters))
    print("Num Points:   " + str(points))
    print("Size (MiB):   " + str(size_bytes / 1024**2))
