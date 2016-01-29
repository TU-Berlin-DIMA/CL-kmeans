#include "kmeans.hpp"

#include <cassert>
#include <algorithm>
#include <limits>

template <typename FP, typename INT>
int cle::KmeansNaive<FP, INT>::initialize() { return 1; }

template <typename FP, typename INT>
int cle::KmeansNaive<FP, INT>::finalize() { return 1; }

template <typename FP, typename INT>
void cle::KmeansNaive<FP, INT>::operator() (
        uint32_t const max_iterations,
        std::vector<FP> const& points_x,
        std::vector<FP> const& points_y,
        std::vector<FP>& centroids_x,
        std::vector<FP>& centroids_y,
        std::vector<INT>& cluster_size,
        std::vector<INT>& memberships,
        KmeansStats& stats) {

    assert(points_x.size() == points_y.size());
    assert(centroids_x.size() == centroids_y.size());
    assert(memberships.size() == points_x.size());
    assert(cluster_size.size() == centroids_x.size());

    bool did_changes;
    uint32_t iterations;

    iterations = 0;
    did_changes = true;
    while (did_changes == true && iterations < max_iterations) {
        did_changes = false;

        // Phase 1: assign points to clusters
        for (INT p = 0; p != points_x.size(); ++p) {
            FP min_distance = std::numeric_limits<FP>::max();
            INT min_centroid;

            for (INT c = 0; c != centroids_x.size(); ++c) {
                FP distance =
                    gaussian_distance(
                            points_x[p], points_y[p],
                            centroids_x[c], centroids_y[c]
                            );
                if (distance < min_distance) {
                    min_distance = distance;
                    min_centroid = c;
                }
            }

            if (min_centroid != memberships[p]) {
                memberships[p] = min_centroid;
                did_changes = true;
            }
        }

        // Phase 2: calculate new clusters
        // Arithmetic mean of all points assigned to cluster
        std::fill(cluster_size.begin(), cluster_size.end(), 0);
        std::fill(centroids_x.begin(), centroids_x.end(), 0);
        std::fill(centroids_y.begin(), centroids_y.end(), 0);

        for (INT p = 0; p != points_x.size(); ++p) {
            INT c = memberships[p];

            cluster_size[c] += 1;
            centroids_x[c] += points_x[p];
            centroids_y[c] += points_y[p];
        }

        for (INT c = 0; c != centroids_x.size(); ++c) {
            centroids_x[c] = centroids_x[c] / cluster_size[c];
            centroids_y[c] = centroids_y[c] / cluster_size[c];
        }

        ++iterations;
    }

    stats.iterations = iterations;
}

template <typename FP, typename INT>
FP cle::KmeansNaive<FP, INT>::gaussian_distance(
        FP a_x, FP a_y, FP b_x, FP b_y) {

    FP t_x = b_x - a_x;
    FP t_y = b_y - a_y;

    return t_x * t_x + t_y * t_y;
}


template class cle::KmeansNaive<float, uint32_t>;
template class cle::KmeansNaive<double, uint64_t>;
