/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef FUSED_FEATURE_SUM_HPP
#define FUSED_FEATURE_SUM_HPP

#include "kernel_path.hpp"

#include "reduce_vector_parcol.hpp"
#include "matrix_binary_op.hpp"

#include "../fused_configuration.hpp"
#include "../measurement/measurement.hpp"
#include "../allocator/readonly_allocator.hpp"
#include "../utility.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <iostream>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>
#include <boost/compute/algorithm/copy.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class FusedFeatureSum {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using ReadonlyVector = boost::compute::vector<T, readonly_allocator<T>>;
    template <typename T>
    using LocalBuffer = boost::compute::local_buffer<T>;

    FusedFeatureSum() :
        g_stride_g_mem_kernel(Utility::log2(MAX_FEATURES)),
        g_stride_l_mem_kernel(Utility::log2(MAX_FEATURES)),
        l_stride_g_mem_kernel(Utility::log2(MAX_FEATURES))
    {}

    void prepare(
            Context context,
            FusedConfiguration config
            )
    {
        static_assert(boost::compute::is_fundamental<PointT>(),
                "PointT must be a boost compute fundamental type");
        static_assert(boost::compute::is_fundamental<LabelT>(),
                "LabelT must be a boost compute fundamental type");
        static_assert(boost::compute::is_fundamental<MassT>(),
                "MassT must be a boost compute fundamental type");
        static_assert(std::is_same<float, PointT>::value
                or std::is_same<double, PointT>::value,
                "PointT must be float or double");

        this->config = config;

        std::string defines;
        defines += " -DCL_INT=uint";
        defines += " -DCL_POINT=";
        defines += boost::compute::type_name<PointT>();
        if (std::is_same<float, PointT>::value) {
            defines += " -DCL_SINT=int";
            defines += " -DCL_POINT_MAX=FLT_MAX";
        }
        else if (std::is_same<double, PointT>::value) {
            defines += " -DCL_SINT=long";
            defines += " -DCL_POINT_MAX=DBL_MAX";
        }
        defines += " -DCL_LABEL=";
        defines += boost::compute::type_name<LabelT>();
        defines += " -DCL_MASS=";
        defines += boost::compute::type_name<MassT>();
        defines += " -DVEC_LEN=";
        defines += std::to_string(this->config.vector_length);

        std::string l_stride_defines = " -DLOCAL_STRIDE";
        std::string g_mem_defines = " -DGLOBAL_MEM";

        for (size_t d = 2; d <= MAX_FEATURES; d = d * 2) {

            std::string features =
                " -DNUM_FEATURES="
                + std::to_string(d);

            Program g_stride_g_mem_program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);

            Program g_stride_l_mem_program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);

            Program l_stride_g_mem_program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);

            size_t kernel_index = Utility::log2(d) - 1;

            try {
                g_stride_g_mem_program.build(defines + features + g_mem_defines);
                g_stride_g_mem_kernel[kernel_index] =
                    g_stride_g_mem_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cerr << g_stride_g_mem_program.build_log() << std::endl;
                throw e;
            }

            try {
                g_stride_l_mem_program.build(defines + features);
                g_stride_l_mem_kernel[kernel_index] =
                    g_stride_l_mem_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cerr << g_stride_l_mem_program.build_log() << std::endl;
                throw e;
            }

            try {
                l_stride_g_mem_program.build(defines + features + l_stride_defines + g_mem_defines);
                l_stride_g_mem_kernel[kernel_index] =
                    l_stride_g_mem_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cerr << l_stride_g_mem_program.build_log() << std::endl;
                throw e;
            }
        }

        reduce_centroids.prepare(context);
        reduce_masses.prepare(context);
        matrix_add_centroids.prepare(context, matrix_add_centroids.Add);
        matrix_add_masses.prepare(context, matrix_add_masses.Add);
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
            )
    {
        return (*this)(
                queue,
                num_features,
                num_points,
                num_clusters,
                points.begin(),
                points.end(),
                centroids.begin(),
                centroids.end(),
                centroids.begin(),
                centroids.end(),
                labels.begin(),
                labels.end(),
                masses.begin(),
                masses.end(),
                datapoint,
                events
                );
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_features,
            size_t num_points,
            size_t num_clusters,
            boost::compute::buffer_iterator<PointT> points_begin,
            boost::compute::buffer_iterator<PointT> points_end,
            boost::compute::buffer_iterator<PointT> old_centroids_begin,
            boost::compute::buffer_iterator<PointT> old_centroids_end,
            boost::compute::buffer_iterator<PointT> new_centroids_begin,
            boost::compute::buffer_iterator<PointT> new_centroids_end,
            boost::compute::buffer_iterator<LabelT> labels_begin,
            boost::compute::buffer_iterator<LabelT> labels_end,
            boost::compute::buffer_iterator<MassT> masses_begin,
            boost::compute::buffer_iterator<MassT> masses_end,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events
            )
    {
        assert(num_features <= MAX_FEATURES);
        assert(points_end - points_begin == (long) (num_points * num_features));
        assert(old_centroids_end - old_centroids_begin == (long) (num_clusters * num_features));
        assert(new_centroids_end - new_centroids_begin == (long) (num_clusters * num_features));
        assert(labels_end - labels_begin == (long) num_points);
        assert(masses_end - masses_begin == (long) num_clusters);
        assert(points_begin.get_index() == 0u);
        assert(old_centroids_begin.get_index() == 0u);
        assert(new_centroids_begin.get_index() == 0u);
        assert(labels_begin.get_index() == 0u);
        assert(masses_begin.get_index() == 0u);

        datapoint.set_name("FusedFeatureSum");

        uint32_t const num_thread_features =
          (this->config.local_size[0] >= num_features)
          ? 1
          : num_features / this->config.local_size[0]
          ;

        size_t const min_centroids_size =
            this->config.global_size[0]
            * num_clusters
            * num_thread_features
            ;
        Vector<PointT> tmp_new_centroids(
                min_centroids_size,
                queue.get_context()
                );

        size_t const min_masses_size =
            this->config.global_size[0]
            * num_clusters;
        Vector<MassT> new_masses(
                min_masses_size,
                queue.get_context()
                );

        LocalBuffer<PointT> local_points(
                this->config.local_size[0]
                * this->config.vector_length
                * num_features
                );
        LocalBuffer<PointT> local_new_centroids(
                this->config.local_size[0]
                * num_thread_features
                * num_clusters
                );
        LocalBuffer<MassT> local_masses(
                this->config.local_size[0] * num_clusters
                );
        LocalBuffer<LabelT> local_labels(
                this->config.local_size[0] * this->config.vector_length
                );

        ReadonlyVector<PointT> ro_centroids(
                num_clusters * num_features,
                queue.get_context()
                );
        boost::compute::copy_async(
                old_centroids_begin,
                old_centroids_end,
                ro_centroids.begin(),
                queue
                );

        size_t kernel_index = Utility::log2(num_features) - 1;
        boost::compute::device device = queue.get_device();
        bool use_local_stride =
            device.type() == device.cpu ||
            device.type() == device.accelerator
            ;
        bool use_local_memory =
            device.type() == device.gpu &&
            device.local_memory_size() >
            (
             local_points.size() * sizeof(PointT) +
             local_new_centroids.size() * sizeof(PointT) +
             local_masses.size() * sizeof(MassT)
            )
            ;
        auto& kernel = (use_local_stride)
            ? l_stride_g_mem_kernel[kernel_index]
            : (
                    use_local_memory
              )
            ? g_stride_l_mem_kernel[kernel_index]
            : g_stride_g_mem_kernel[kernel_index]
            ;

        if (use_local_memory) {
            kernel.set_args(
                    points_begin.get_buffer(),
                    ro_centroids,
                    tmp_new_centroids,
                    new_masses,
                    labels_begin.get_buffer(),
                    local_points,
                    local_new_centroids,
                    local_masses,
                    local_labels,
                    (cl_uint)num_points,
                    (cl_uint)num_clusters,
                    (cl_uint)num_thread_features
                );
        }
        else {
            kernel.set_args(
                    points_begin.get_buffer(),
                    ro_centroids,
                    tmp_new_centroids,
                    new_masses,
                    labels_begin.get_buffer(),
                    local_labels,
                    (cl_uint)num_points,
                    (cl_uint)num_clusters,
                    (cl_uint)num_thread_features
                );
        }

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                kernel,
                1,
                work_offset,
                this->config.global_size,
                this->config.local_size,
                events);

        datapoint.add_event() = event;

        boost::compute::wait_list wait_list;
        wait_list.insert(event);

        size_t num_tiles =
            this->config.global_size[0]
            * num_thread_features
            / num_features
            ;

        event = reduce_centroids(
                queue,
                num_tiles,
                num_clusters * num_features,
                tmp_new_centroids,
                datapoint.create_child(),
                wait_list
                );

        wait_list.insert(event);

        event = matrix_add_centroids.matrix(
                queue,
                num_features,
                num_clusters,
                new_centroids_begin,
                new_centroids_end,
                tmp_new_centroids.begin(),
                tmp_new_centroids.begin() + num_clusters * num_features,
                datapoint.create_child(),
                wait_list
                );

        wait_list.insert(event);

        event = reduce_masses(
                queue,
                this->config.global_size[0],
                num_clusters,
                new_masses,
                datapoint.create_child(),
                wait_list
                );

        wait_list.insert(event);

        event = matrix_add_masses.matrix(
                queue,
                1,
                num_clusters,
                masses_begin,
                masses_end,
                new_masses.begin(),
                new_masses.begin() + num_clusters,
                datapoint.create_child(),
                wait_list
                );

        wait_list.insert(event);

        return event;
    }


private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_fused_feature_sum.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_fused_feature_sum";
    static constexpr const size_t MAX_FEATURES = 1024;

    std::vector<Kernel> g_stride_g_mem_kernel;
    std::vector<Kernel> g_stride_l_mem_kernel;
    std::vector<Kernel> l_stride_g_mem_kernel;
    FusedConfiguration config;
    ReduceVectorParcol<PointT> reduce_centroids;
    ReduceVectorParcol<MassT> reduce_masses;
    MatrixBinaryOp<PointT, PointT> matrix_add_centroids;
    MatrixBinaryOp<MassT, MassT> matrix_add_masses;
};

}


#endif /* FUSED_FEATURE_SUM_HPP */
