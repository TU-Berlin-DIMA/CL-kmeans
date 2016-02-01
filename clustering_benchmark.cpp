#include "clustering_benchmark.hpp"

#include <cstdint>
#include <algorithm> // std::equal


cle::ClusteringBenchmarkStats::ClusteringBenchmarkStats(const uint32_t num_runs)
    :
        microseconds(num_runs),
        kmeans_stats(num_runs),
        num_runs_(num_runs)
{}

void cle::ClusteringBenchmarkStats::print_times() {
    std::cout << this->num_runs_ << " runs, in µs: [";
    for (uint32_t r = 0; r < microseconds.size(); ++r) {
        std::cout << microseconds[r];
        if (r != microseconds.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT>::ClusteringBenchmark(
        const uint32_t num_runs,
        const INT num_points,
        const uint32_t max_iterations,
        std::vector<FP, AllocFP>&& points_x,
        std::vector<FP, AllocFP>&& points_y
        )
    :
        num_runs_(num_runs),
        num_points_(num_points),
        num_clusters_(0),
        max_iterations_(max_iterations),
        points_x_(std::move(points_x)),
        points_y_(std::move(points_y)),
        memberships_(num_points)
{}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT>::initialize(
        const INT num_clusters,
        InitCentroidsFunction init_centroids
        ) {

    num_clusters_ = num_clusters;
    init_centroids_ = init_centroids;

    centroids_x_.resize(num_clusters);
    centroids_y_.resize(num_clusters);
    cluster_size_.resize(num_clusters);

    return 1;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT>::finalize() {
    return 1;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
cle::ClusteringBenchmarkStats cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT>::run(
        ClusteringFunction f) {

    cle::Timer timer;
    ClusteringBenchmarkStats bs(this->num_runs_);

    for (uint32_t r = 0; r < this->num_runs_; ++r) {
        init_centroids_(
                points_x_, points_y_,
                centroids_x_, centroids_y_
                );

        timer.start();
        f(
                max_iterations_,
                points_x_, points_y_,
                centroids_x_,
                centroids_y_,
                cluster_size_,
                memberships_,
                bs.kmeans_stats[r]
         );
        bs.microseconds[r] = timer.stop<std::chrono::microseconds>();
    }

    return bs;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT>::setVerificationReference(
        ClusteringFunction ref) {

    cle::KmeansStats stats;

    reference_memberships_.resize(num_points_);

    ref(
            max_iterations_,
            points_x_, points_y_,
            centroids_x_, centroids_y_,
            cluster_size_,
            reference_memberships_,
            stats
       );

    return 1;
}

template<typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT>::verify(ClusteringFunction f) {

    cle::KmeansStats stats;
    int is_correct;

    f(
            max_iterations_,
            points_x_, points_y_,
            centroids_x_, centroids_y_,
            cluster_size_,
            memberships_,
            stats
       );

    is_correct = std::equal(
            reference_memberships_.begin(),
            reference_memberships_.end(),
            memberships_.begin());

    return is_correct;
}

template class cle::ClusteringBenchmark<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>>;
template class cle::ClusteringBenchmark<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>>;
template class cle::ClusteringBenchmark<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32>;
template class cle::ClusteringBenchmark<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64>;
