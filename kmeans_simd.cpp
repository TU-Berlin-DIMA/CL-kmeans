#include "kmeans.hpp"

#include <cassert>
#include <algorithm>
#include <limits>
#include <x86intrin.h>

__m256 gaussian_distance_simd(
        __m256 a_x, __m256 a_y,
        __m256 b_x, __m256 b_y) {

    __m256 t_x = _mm256_sub_ps(b_x, a_x);
    __m256 t_y = _mm256_sub_ps(b_y, a_y);

    t_x = _mm256_mul_ps(t_x, t_x);
    t_y = _mm256_mul_ps(t_y, t_y);

    return _mm256_add_ps(t_x, t_y);
}

float gaussian_distance(
        float a_x, float a_y,
        float b_x, float b_y
        ) {

    float t_x = b_x - a_x;
    float t_y = b_y - a_y;

    return t_x * t_x + t_y * t_y;
}

int cle::KmeansSIMD32::initialize() {
    return 1;
}

int cle::KmeansSIMD32::finalize() {
    return 1;
}

void cle::KmeansSIMD32::operator() (
    uint32_t const max_iterations,
    std::vector<float, AlignedAllocatorFP32> const& points_x,
    std::vector<float, AlignedAllocatorFP32> const& points_y,
    std::vector<float, AlignedAllocatorFP32>& centroids_x,
    std::vector<float, AlignedAllocatorFP32>& centroids_y,
    std::vector<uint32_t, AlignedAllocatorINT32>& cluster_size,
    std::vector<uint32_t, AlignedAllocatorINT32>& memberships,
    KmeansStats& stats) {

    assert(points_x.size() == points_y.size());
    assert(centroids_x.size() == centroids_y.size());
    assert(memberships.size() == points_x.size());
    assert(cluster_size.size() == centroids_x.size());

    const uint32_t num_points = points_x.size();
    const uint32_t num_clusters = centroids_x.size();
    constexpr uint32_t vec_size = 8;

    int did_changes = true;
    uint32_t iterations = 0;
    while (did_changes != 0 && iterations < max_iterations) {
        did_changes = 0;

        for (uint32_t p = 0; p < num_points; p += vec_size) {
            __m256 min_distance;
            __m256i min_centroid_id;

            __m256 point_x = _mm256_load_ps(&points_x[p]);
            __m256 point_y = _mm256_load_ps(&points_y[p]);

            min_distance = _mm256_set1_ps(std::numeric_limits<float>::max());

            for (uint32_t c = 0; c != num_clusters; ++c) {
                __m256i centroid_id = _mm256_set1_epi32(c);
                __m256 centroid_x = _mm256_set1_ps(centroids_x[c]);
                __m256 centroid_y = _mm256_set1_ps(centroids_y[c]);

                __m256 distance = gaussian_distance_simd(
                        point_x, point_y,
                        centroid_x, centroid_y
                        );

                __m256 min_mask = _mm256_cmp_ps(distance, min_distance, _CMP_LT_OS);
                did_changes |= _mm256_movemask_ps(min_mask);
                min_distance = _mm256_blendv_ps(min_distance, distance, min_mask);
                min_centroid_id = _mm256_blendv_ps(min_centroid_id, centroid_id, min_mask);
            }

            _mm256_store_si256((__m256i*)&memberships[p], min_centroid_id);
        }

        uint32_t remain;
        if ((remain = num_points % vec_size) != 0) {
            for (uint32_t p = num_points - remain; p < num_points; ++p) {
                float min_distance = std::numeric_limits<float>::max();
                uint32_t min_centroid_id;

                for (uint32_t c = 0; c != num_clusters; ++c) {
                    float distance = gaussian_distance(
                            points_x[p], points_y[p],
                            centroids_x[c], centroids_y[c]
                            );

                    if (distance < min_distance) {
                        min_distance = distance;
                        min_centroid_id = c;
                        did_changes |= 1;
                    }
                }

                memberships[p] = min_centroid_id;
            }
        }

        std::fill(cluster_size.begin(), cluster_size.end(), 0);
        std::fill(centroids_x.begin(), centroids_x.end(), 0);
        std::fill(centroids_y.begin(), centroids_y.end(), 0);

        for (uint32_t p = 0; p != num_points; ++p) {
            uint32_t c = memberships[p];

            cluster_size[c] += 1;
            centroids_x[c] += points_x[p];
            centroids_y[c] += points_y[p];
        }

        for (uint32_t c = 0; c != num_clusters; ++c) {
            centroids_x[c] = centroids_x[c] / cluster_size[c];
            centroids_y[c] = centroids_y[c] / cluster_size[c];
        }

        ++iterations;
    }

    stats.iterations = iterations;
}
