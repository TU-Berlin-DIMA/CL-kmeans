#ifndef MASS_UPDATE_GLOBAL_ATOMIC_HPP
#define MASS_UPDATE_GLOBAL_ATOMIC_HPP

#include "kernel_path.hpp"

#include "../mass_update_configuration.hpp"
#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename LabelT, typename MassT>
class MassUpdateGlobalAtomic {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using Vector = boost::compute::vector<T>;

    void prepare(
            Context context,
            MassUpdateConfiguration config) {
        this->config = config;

        std::string defines;
        if (std::is_same<cl_uint, LabelT>::value) {
            defines = "-DTYPE32";
        }
        else if (std::is_same<cl_ulong, LabelT>::value) {
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
            size_t num_points,
            size_t num_clusters,
            Vector<LabelT>& labels,
            Vector<MassT>& masses,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events
            ) {

        datapoint.set_name("MassUpdateGlobalAtomic");

        this->kernel.set_args(
                labels,
                masses,
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
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("histogram_global.cl");
    static constexpr const char* KERNEL_NAME = "histogram_global";

    Kernel kernel;
    MassUpdateConfiguration config;
};


}

#endif /* MASS_UPDATE_GLOBAL_ATOMIC_HPP */
