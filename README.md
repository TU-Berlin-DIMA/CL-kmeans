# CL k-Means

CL k-Means is an efficient and portable implementation of Lloyd's k-Means
algorithm in OpenCL. It introduces a more efficient execution strategy that
requires only a single pass over data. This single pass optimization is based
on a new centroid update algorithm that features a reduced cache footprint.

## Build Dependencies

- A working OpenCL installation. Refer to [Andreas Klöckner's wiki](https://wiki.tiker.net/OpenCLHowTo).
- Clang 3.8 or greater
- Boost version 1.61 or greater

## Build Instructions

```
git clone --recursive https://github.com/TU-Berlin-DIMA/CL-kmeans.git
mkdir CL-kmeans/build
cd CL-kmeans/build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j`grep processor /proc/cpuinfo | wc -l`
```

## Usage

First, generate binary files in a simple binary file format. Then run the benchmark. Finally, view the total OpenCL kernel runtimes stored in the output CSV file.

```
./scripts/generate_features.py
./bench --csv runtime.csv --config ../configurations/intel_core_i7-6700K_three_stage.conf ../data/cluster_data_4f_10c_2048mb.bin
grep TotalTime *runtime_mnts.csv # Total runtime in microseconds in last column
```

## Configurations

CL k-Means can be tuned to different types of processors using simple
configuration files. Each file consists of a key-value pairs that define the
execution strategy and hardware tuning options.

See example configurations for Intel Core i7-6700K and Nvidia GeForce GTX 1080
processors in the '/configurations' directory.

## Publications

This project has resulted in the following academic publications:

[C. Lutz et al., "Efficient and Scalable k-Means on GPUs", in Datenbanken Spektrum 2018](https://doi.org/10.1007/s13222-018-0293-x)  
[C. Lutz et al., "Efficient k-Means on GPUs", in DaMoN 2018](http://doi.acm.org/10.1145/3211922.3211925)

To cite these works, add these BibTeX snippets to your bibliography:

```
@article{lutz:dbspektrum:2018,
  author    = {Clemens Lutz and
               Sebastian Bre{\ss} and
               Tilmann Rabl and
               Steffen Zeuch and
               Volker Markl},
  title     = {Efficient and Scalable k-Means on GPUs},
  journal   = {Datenbank-Spektrum},
  volume    = {18},
  number    = {3},
  pages     = {157--169},
  year      = {2018},
  url       = {https://doi.org/10.1007/s13222-018-0293-x},
  doi       = {10.1007/s13222-018-0293-x}
}
```

```
@inproceedings{lutz:damon:2018,
  author    = {Clemens Lutz and
               Sebastian Bre{\ss} and
               Tilmann Rabl and
               Steffen Zeuch and
               Volker Markl},
  title     = {Efficient k-Means on GPUs},
  booktitle = {Proceedings of the 14th International Workshop on Data Management
               on New Hardware, Houston, TX, USA, June 11, 2018},
  pages     = {3:1--3:3},
  year      = {2018},
  url       = {https://doi.org/10.1145/3211922.3211925},
  doi       = {10.1145/3211922.3211925}
}
```
