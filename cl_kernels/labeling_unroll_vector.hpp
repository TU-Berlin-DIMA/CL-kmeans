#ifndef LABELING_UNROLL_VECTOR_HPP
#define LABELING_UNROLL_VECTOR_HPP

#include "kernel_path.hpp"

#include "../temp.hpp"
#include "../labeling_configuration.hpp"

#include <cassert>
#include <string>
#include <type_traits>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, bool ColMajor>
class LabelingUnrollVector {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;

    void prepare(Context context, LabelingConfiguration config) {
        this->config = config;

        std::string defines;
        if (std::is_same<cl_float, PointT>::value) {
            defines = "-DTYPE32";
        }
        else if (std::is_same<cl_double, PointT>::value) {
            defines = "-DTYPE64";
        }
        else {
            assert(false);
        }

        defines += " -DVEC_LEN=" + std::to_string(this->config.vector_length);
        defines += " -DCLUSTERS_UNROLL=" + std::to_string(this->config.unroll_clusters_length);
        defines += " -DFEATURES_UNROLL=" + std::to_string(this->config.unroll_features_length);

        Program program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);

        program.build(defines);

        this->kernel = program.create_kernel(KERNEL_NAME);

    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_features,
            size_t num_points,
            size_t num_clusters,
            boost::compute::vector<int>& did_changes,
            boost::compute::vector<const PointT>& points,
            boost::compute::vector<PointT>& centroids,
            boost::compute::vector<LabelT>& labels,
            Clustering::MeasurementLogger&,
            boost::compute::wait_list const& events) {

        this->kernel.set_args(
                did_changes,
                points,
                centroids,
                labels,
                num_features,
                num_points,
                num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        return queue.enqueue_nd_range_kernel(
                this->kernel,
                1,
                work_offset,
                this->config.global_size,
                this->config.local_size,
                events);
    }

private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_labeling_vp_clc.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_labeling_vp_clc";

    Kernel kernel;
    LabelingConfiguration config;

};

}

#endif /* LABELING_UNROLL_VECTOR_HPP */
