#ifndef CENTROID_UPDATE_FEATURE_SUM_HPP
#define CENTROID_UPDATE_FEATURE_SUM_HPP

#include "kernel_path.hpp"

#include "../centroid_update_configuration.hpp"
#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class CentroidUpdateFeatureSum {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using LocalBuffer = boost::compute::local_buffer<T>;

    void prepare(
            Context context,
            CentroidUpdateConfiguration config) {
        this->config = config;

        std::string defines;
        if (std::is_same<float, PointT>::value) {
            defines = "-DTYPE32";
        }
        else if (std::is_same<double, PointT>::value) {
            defines = "-DTYPE64";
        }
        else {
            assert(false);
        }

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
            Vector<PointT>& points,
            Vector<PointT>& centroids,
            Vector<LabelT>& labels,
            Vector<MassT>& masses,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events
            ) {

        assert(points.size() == num_points * num_features);
        assert(labels.size() == num_points);
        assert(centroids.size() == num_clusters * num_features);
        assert(masses.size() >= num_clusters);

        datapoint.set_name("CentroidUpdateFeatureSum");

        LocalBuffer<PointT> local_centroids(
                num_clusters * this->config.local_size[0]);
        LocalBuffer<PointT> local_points(
                this->config.local_size[0] * this->config.local_size[0]);

        this->kernel.set_args(
                points,
                centroids,
                masses,
                labels,
                local_centroids,
                local_points,
                num_features,
                num_points,
                num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                this->kernel,
                1,
                work_offset,
                this->config.global_size,
                this->config.local_size,
                events);
        datapoint.add_event() = event;
        return event;
    }


private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_feature_sum.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_feature_sum";

    Kernel kernel;
    CentroidUpdateConfiguration config;
};

}


#endif /* CENTROID_UPDATE_FEATURE_SUM_HPP */
