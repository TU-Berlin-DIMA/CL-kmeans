#include "kmeans.hpp"

#include "cl_common.hpp"

#include "kmeans_cl_api.hpp"

#include <cstdint>
#include <cstddef> // size_t
#include <vector>
#include <algorithm> // std::any_of

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

cle::Kmeans_GPU::Kmeans_GPU(cl::Context const& context, cl::CommandQueue const& queue) :
    context_(context), queue_(queue)
{}


int cle::Kmeans_GPU::initialize() {

    cl_int err = CL_SUCCESS;
    cl_uint work_item_dims;

    // Create kernel
    err = this->kmeans_kernel_.initialize(this->context_);
    if (err != CL_SUCCESS) {
        return err;
    }

    cl::Device device;
    this->queue_.getInfo(CL_QUEUE_DEVICE, &device);

    device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &work_item_dims);
    this->max_work_item_sizes_.resize(work_item_dims);
    device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, this->max_work_item_sizes_.data());
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &this->max_work_group_size_);

    return 1;
}

int cle::Kmeans_GPU::finalize() {
    return 1;
}

int cle::Kmeans_GPU::operator() (
        uint32_t const max_iterations,
        std::vector<double> const& points_x,
        std::vector<double> const& points_y,
        std::vector<double>& centroids_x,
        std::vector<double>& centroids_y,
        std::vector<uint64_t>& cluster_size,
        std::vector<uint64_t>& cluster_assignment
        ) {

    uint32_t iterations;
    bool did_changes;

    size_t global_size = this->max_work_group_size_ * this->max_work_item_sizes_[0];
    size_t num_points = points_x.size();
    size_t num_clusters = centroids_x.size();
    std::vector<cl_char> h_did_changes(global_size);

    cle::TypedBuffer<cl_char> d_did_changes(this->context_, CL_MEM_READ_WRITE, global_size);
    cle::TypedBuffer<cl_double> d_points_x(this->context_, CL_MEM_READ_ONLY, num_points);
    cle::TypedBuffer<cl_double> d_points_y(this->context_, CL_MEM_READ_ONLY, num_points);
    cle::TypedBuffer<cl_double> d_point_distance(this->context_, CL_MEM_READ_WRITE, num_points);
    cle::TypedBuffer<cl_double> d_centroids_x(this->context_, CL_MEM_READ_WRITE, num_clusters);
    cle::TypedBuffer<cl_double> d_centroids_y(this->context_, CL_MEM_READ_WRITE, num_clusters);
    cle::TypedBuffer<cl_ulong> d_cluster_size(this->context_, CL_MEM_READ_WRITE, num_clusters);
    cle::TypedBuffer<cl_ulong> d_cluster_assignment(this->context_, CL_MEM_READ_WRITE, num_points);


    // copy points (x,y) host -> device
    cle_sanitize_val_return(
            this->queue_.enqueueWriteBuffer(
                d_points_x,
                CL_FALSE,
                0,
                d_points_x.bytes(),
                points_x.data()
                ));

    cle_sanitize_val_return(
            this->queue_.enqueueWriteBuffer(
                d_points_y,
                CL_FALSE,
                0,
                d_points_y.bytes(),
                points_y.data()
                ));

    cle_sanitize_val(
            this->queue_.enqueueFillBuffer(
                d_point_distance,
                __DBL_MAX__,
                0,
                d_point_distance.bytes()
                ));

    cle_sanitize_val_return(
            this->queue_.enqueueWriteBuffer(
                d_centroids_x,
                CL_FALSE,
                0,
                d_centroids_x.bytes(),
                centroids_x.data()
                ));

    cle_sanitize_val_return(
            this->queue_.enqueueWriteBuffer(
                d_centroids_y,
                CL_FALSE,
                0,
                d_centroids_y.bytes(),
                centroids_y.data()
                ));

    cle_sanitize_val_return(
            this->queue_.enqueueWriteBuffer(
                d_cluster_size,
                CL_FALSE,
                0,
                d_cluster_size.bytes(),
                cluster_size.data()
                ));

    cle_sanitize_val_return(
            this->queue_.enqueueWriteBuffer(
                d_cluster_assignment,
                CL_FALSE,
                0,
                d_cluster_assignment.bytes(),
                cluster_assignment.data()
                ));

    // copy centroids (x,y) host -> device

    iterations = 0;
    did_changes = true;
    while (did_changes && iterations < max_iterations) {

        // set did_changes to false on device
        cle_sanitize_val(
                this->queue_.enqueueFillBuffer(
                    d_did_changes,
                    false,
                    0,
                    d_did_changes.bytes()
                    ));

        // execute kernel
        kmeans_kernel_(
                cl::EnqueueArgs(
                    this->queue_,
                    cl::NDRange(global_size),
                    cl::NDRange(this->max_work_group_size_)
                    ),
                d_did_changes,
                d_points_x,
                d_points_y,
                d_point_distance,
                d_centroids_x,
                d_centroids_y,
                d_cluster_size,
                d_cluster_assignment
                );

        // copy did_changes device -> host
        cle_sanitize_val(
                this->queue_.enqueueReadBuffer(
                    d_did_changes,
                    CL_TRUE,
                    0,
                    d_did_changes.bytes(),
                    h_did_changes.data()
                    ));

        // inspect did_changes
        did_changes = std::any_of(
                h_did_changes.cbegin(),
                h_did_changes.cend(),
                [](cl_char i){ return i == CL_TRUE; }
                );

        // flip old_centroids <-> centroids

        ++iterations;
    }

    // copy centroids (x,y) device -> host
    // copy cluster sizes device -> host
    // copy point assignments device -> host

    return 1;
}
