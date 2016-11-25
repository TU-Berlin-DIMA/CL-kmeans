/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include <cl_kernels/lloyd_feature_sum_api.hpp>
#include <cl_kernels/lloyd_merge_sum_api.hpp>
#include <cl_kernels/lloyd_feature_sum_pardim_api.hpp>
#include <cl_kernels/reduce_vector_parcol_api.hpp>
#include <measurement/measurement.hpp>
#include <matrix.hpp>
#include <cluster_generator.hpp>

#include <gtest/gtest.h>

#include <random>
#include <vector>
#include <algorithm>
#include <measurement/measurement.hpp>

#include "opencl_setup.hpp"

#define MEGABYTE (1024ull * 1024)
#define GIGABYTE (1024ull * 1024 * 1024)

using Matrix32 = cle::Matrix<float, std::allocator<float>, uint32_t>;

/*
 * Calculate verification data for the feature sum
 *
 * points:      input PxF matrix of data points
 * centroids:   output CxF matrix of centroids
 * mass:        output C vector of cluster masses
 *                  (aka size of cluster)
 * labels:      input P vector of point labels
 *                  (aka cluster the point belogns to)
 */
void feature_sum_verify(
        Matrix32 const& points,
        Matrix32& centroids,
        std::vector<uint32_t>& mass,
        std::vector<uint32_t> const& labels
        ) {

    std::fill(centroids.begin(), centroids.end(), 0);
    std::fill(mass.begin(), mass.end(), 0);

    for (size_t f = 0; f < centroids.cols(); ++f) {
        for (size_t p = 0; p < points.rows(); ++p) {
            uint32_t label = labels[p];
            centroids(label, f) += points(p, f);
            ++mass[label];
        }
    }

    for (size_t f = 0; f < centroids.cols(); ++f) {
        for (size_t c = 0; c < centroids.rows(); ++c) {
            centroids(c, f) /= mass[c];
        }
    }
}

class AbstractFeatureSum {
public:
    void set_cl_dimensions(size_t global_size, size_t local_size) {
        global_size_ = global_size;
        local_size_ = local_size;
    }

    void virtual test(
            Matrix32 const& points,
            Matrix32& centroids,
            std::vector<uint32_t> const& mass,
            std::vector<uint32_t> const& labels
            ) = 0;

    void virtual performance(
            Matrix32 const& points,
            std::vector<uint32_t> const& mass,
            std::vector<uint32_t> const& labels,
            uint32_t num_runs,
            Measurement::Measurement& measurement
            ) = 0;

protected:
    size_t global_size_ = 32 * 8;
    size_t local_size_ = 32;
};

template <typename Kernel, Measurement::DataPointType::t point_type>
class FeatureSum : public AbstractFeatureSum {
public:
    void virtual test(
            Matrix32 const& points,
            Matrix32& centroids,
            std::vector<uint32_t> const& mass,
            std::vector<uint32_t> const& labels
            ) {

        Kernel fsum_kernel;
        fsum_kernel.initialize(clenv->context);
        cl::Event fsum_event;

        cle::ReduceVectorParcolAPI<cl_float, cl_uint> reduce_kernel;
        reduce_kernel.initialize(clenv->context);
        cl::Event reduce_event;

        size_t num_centroid_blocks = fsum_kernel.get_num_global_blocks(
                global_size_,
                local_size_,
                centroids.cols(),
                centroids.rows()
                );
        size_t d_centroids_size = centroids.size() * num_centroid_blocks;

        cle::TypedBuffer<cl_float> d_points(
                clenv->context,
                CL_MEM_READ_WRITE,
                points.size()
                );
        cle::TypedBuffer<cl_float> d_centroids(
                clenv->context,
                CL_MEM_READ_WRITE,
                d_centroids_size
                );
        cle::TypedBuffer<cl_uint> d_mass(
                clenv->context,
                CL_MEM_READ_WRITE,
                mass.size()
                );
        cle::TypedBuffer<cl_uint> d_labels(
                clenv->context,
                CL_MEM_READ_WRITE,
                labels.size()
                );

        clenv->queue.enqueueWriteBuffer(
                d_points,
                CL_FALSE,
                0,
                d_points.bytes(),
                points.data()
                );

        clenv->queue.enqueueWriteBuffer(
                d_mass,
                CL_FALSE,
                0,
                d_mass.bytes(),
                mass.data()
                );

        clenv->queue.enqueueWriteBuffer(
                d_labels,
                CL_FALSE,
                0,
                d_labels.bytes(),
                labels.data()
                );

        fsum_kernel(
                cl::EnqueueArgs(
                    clenv->queue,
                    cl::NDRange(global_size_),
                    cl::NDRange(local_size_)),
                points.cols(),
                points.rows(),
                centroids.rows(),
                d_points,
                d_centroids,
                d_mass,
                d_labels,
                fsum_event
              );

        reduce_kernel(
                cl::EnqueueArgs(
                    clenv->queue,
                    cl::NullRange,
                    cl::NullRange),
                num_centroid_blocks,
                centroids.size(),
                d_centroids,
                reduce_event
                );

        clenv->queue.enqueueReadBuffer(
                d_centroids,
                CL_TRUE,
                0,
                centroids.size() * sizeof(cl_float),
                centroids.data()
                );
    }

    void virtual performance(
            Matrix32 const& points,
            std::vector<uint32_t> const& mass,
            std::vector<uint32_t> const& labels,
            uint32_t num_runs,
            Measurement::Measurement& measurement
            ) {
        Kernel fsum_kernel;
        fsum_kernel.initialize(clenv->context);
        cl::Event fsum_event;

        size_t num_features = points.cols();
        size_t num_clusters = mass.size();

        size_t num_centroid_blocks = fsum_kernel.get_num_global_blocks(
                global_size_,
                local_size_,
                num_features,
                num_clusters
                );
        size_t d_centroids_size = num_features * num_clusters * num_centroid_blocks;

        cle::TypedBuffer<cl_float> d_points(
                clenv->context,
                CL_MEM_READ_WRITE,
                points.size()
                );
        cle::TypedBuffer<cl_float> d_centroids(
                clenv->context,
                CL_MEM_READ_WRITE,
                d_centroids_size
                );
        cle::TypedBuffer<cl_uint> d_mass(
                clenv->context,
                CL_MEM_READ_WRITE,
                mass.size()
                );
        cle::TypedBuffer<cl_uint> d_labels(
                clenv->context,
                CL_MEM_READ_WRITE,
                labels.size()
                );

        clenv->queue.enqueueWriteBuffer(
                d_points,
                CL_FALSE,
                0,
                d_points.bytes(),
                points.data()
                );

        clenv->queue.enqueueWriteBuffer(
                d_mass,
                CL_FALSE,
                0,
                d_mass.bytes(),
                mass.data()
                );

        clenv->queue.enqueueWriteBuffer(
                d_labels,
                CL_FALSE,
                0,
                d_labels.bytes(),
                labels.data()
                );

        for (uint32_t r = 0; r < num_runs; ++r) {
            fsum_kernel(
                    cl::EnqueueArgs(
                        clenv->queue,
                        cl::NDRange(global_size_),
                        cl::NDRange(local_size_)),
                    num_features,
                    points.rows(),
                    num_clusters,
                    d_points,
                    d_centroids,
                    d_mass,
                    d_labels,
                    measurement.add_datapoint(r).add_opencl_event()
                    );
        }
    }
};

class UniformDistribution :
    public ::testing::TestWithParam<
    std::tuple<
    size_t, // num_features
    size_t, // num_clusters
    size_t, // global_size
    size_t, // local_size
    std::shared_ptr<AbstractFeatureSum>
    >>
{
protected:
    virtual void SetUp() {
        size_t num_features, num_clusters, global_size, local_size;
        std::shared_ptr<AbstractFeatureSum> fsum;
        std::tie(num_features, num_clusters, global_size, local_size, fsum) = GetParam();

        if (points.cols() != num_features) {
            cle::ClusterGenerator clugen;
            clugen.total_size(points_bytes);
            clugen.num_features(num_features);
            clugen.point_multiple(point_multiple);
            clugen.num_clusters(10);
            clugen.domain(-100.0, 100.0);
            clugen.cluster_radius(10.0);

            Matrix32 centroids_ground_truth; // discarded as not sum
            std::vector<uint32_t> labels_ground_truth;
            clugen.generate_matrix(
                    points,
                    centroids_ground_truth,
                    labels_ground_truth);
        }

        if (verify_centroids.rows() != num_clusters
                || verify_centroids.cols() != num_features) {

            verify_centroids.resize(num_clusters, num_features);
            mass.resize(num_clusters);

            labels.resize(points.rows());
            std::default_random_engine rgen;
            std::uniform_int_distribution<uint32_t> uniform(0, num_clusters - 1);
            std::generate(
                    labels.begin(),
                    labels.end(),
                    [&](){ return uniform(rgen); }
                    );

            feature_sum_verify(
                    points,
                    verify_centroids,
                    mass,
                    labels);
        }
    }

    virtual void TearDown() {
    }

    // static constexpr size_t points_bytes = 64 * MEGABYTE;
    static constexpr size_t points_bytes = 1024;
    static constexpr size_t point_multiple = 32;
    static Matrix32 points, verify_centroids;
    static std::vector<uint32_t> labels, mass;
};
Matrix32 UniformDistribution::points;
Matrix32 UniformDistribution::verify_centroids;
std::vector<uint32_t> UniformDistribution::labels;
std::vector<uint32_t> UniformDistribution::mass;

TEST_P(UniformDistribution, Test) {
    std::shared_ptr<AbstractFeatureSum> fsum;
    size_t num_features, num_clusters, global_size, local_size;
    std::tie(num_features, num_clusters, global_size, local_size, fsum) = GetParam();

    cle::Matrix<float, std::allocator<float>, uint32_t> test_centroids;
    test_centroids.resize(num_clusters, num_features);

    fsum->set_cl_dimensions(global_size, local_size);
    fsum->test(points, test_centroids, mass, labels);

    ASSERT_TRUE(test_centroids.size() == verify_centroids.size());
    for (size_t i = 0; i < test_centroids.size(); ++i) {
        EXPECT_FLOAT_EQ(verify_centroids.data()[i], test_centroids.data()[i]);
    }
}

TEST_P(UniformDistribution, Performance) {

    const size_t num_runs = 5;

    std::shared_ptr<AbstractFeatureSum> fsum;
    size_t num_features, num_clusters, global_size, local_size;
    std::tie(num_features, num_clusters, global_size, local_size, fsum) = GetParam();

    Measurement::Measurement measurement;
    measurement_setup(measurement, clenv->device, num_runs);
    measurement.set_parameter(
            Measurement::ParameterType::NumFeatures,
            std::to_string(num_features));
    measurement.set_parameter(
            Measurement::ParameterType::NumPoints,
            std::to_string(points.rows()));
    measurement.set_parameter(
            Measurement::ParameterType::NumClusters,
            std::to_string(num_clusters));
    measurement.set_parameter(
            Measurement::ParameterType::IntType,
            "uint32_t");
    measurement.set_parameter(
            Measurement::ParameterType::FloatType,
            "float");
    measurement.set_parameter(
            Measurement::ParameterType::CLLocalSize,
            std::to_string(local_size));
    measurement.set_parameter(
            Measurement::ParameterType::CLGlobalSize,
            std::to_string(global_size));

    cle::Matrix<float, std::allocator<float>, uint32_t> test_centroids;
    test_centroids.resize(num_clusters, num_features);

    fsum->set_cl_dimensions(global_size, local_size);
    fsum->performance(points, mass, labels, num_runs, measurement);

    measurement.write_csv("feature_sum.csv");

    SUCCEED();
}

INSTANTIATE_TEST_CASE_P(
        StandardParameters,
        UniformDistribution,
        ::testing::Combine(
            ::testing::Values(2, 4),
            ::testing::Values(2, 4),
            ::testing::Values((32 * 4 * 64), (32 * 4 * 64 * 32)),
            ::testing::Values(32, 64, 128, 256),
            ::testing::Values(
                new FeatureSum<cle::LloydMergeSumAPI<cl_float, cl_uint>, Measurement::DataPointType::LloydCentroidsMergeSum>,
                new FeatureSum<cle::LloydFeatureSumPardimAPI<cl_float, cl_uint>, Measurement::DataPointType::LloydCentroidsFeatureSumPardim>
                )
            ));

int main (int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    clenv = new CLEnvironment;
    ::testing::AddGlobalTestEnvironment(clenv);
    return RUN_ALL_TESTS();
}
