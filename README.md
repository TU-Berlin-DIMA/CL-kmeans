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

Generated binary files in simple file format.

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

[C. Lutz et al., "Efficient and Scalable k-Means on GPUs", in Datenbanken Spektrum 2018](https://doi.org/10.1007/s13222-018-0293-x)  
[C. Lutz et al., "Efficient k-Means on GPUs", in DaMoN 2018](http://doi.acm.org/10.1145/3211922.3211925)
