#!/usr/bin/env python

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
# 
# 
# Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>

import subprocess
import os

generator = "build/generator"

outdir = "data"
stem = "cluster_data_"
postfix = "f_10c_2048mb.bin"
features = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
clusters = 10
divisor = 32
size = 2048

if not os.path.exists(outdir):
    os.mkdir(outdir)

args = [generator]
args.extend(["--clusters", str(clusters)])
args.extend(["--divisor", str(divisor)])
args.extend(["--size", str(size)])
for s in features:
    file_name = outdir + "/" + stem + str(s) + postfix
    print("Generatring " + file_name)
    subprocess.call(args + ["--features", str(s), file_name])
