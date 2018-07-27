/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2018, Lutz, Clemens <lutzcle@cml.li>
 */

#include <measurement/measurement.hpp>
#include <simple_buffer_cache.hpp>
#include <single_device_scheduler.hpp>

#include <boost/compute/core.hpp>
#include <boost/program_options.hpp>

#include <vector>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

#include <sys/mman.h>
#include <unistd.h>

namespace bc = boost::compute;
namespace po = boost::program_options;

constexpr char zero_source[] =
R"ENDSTR(
__kernel void zero(__global int * const restrict buffer, uint size)
{
    for (uint i = get_global_id(0); i < size; i += get_global_size(0)) {
        buffer[i] = 0;
    }
}
)ENDSTR";

class TransferBench {
public:
    TransferBench(
            bc::device device,
            bc::command_queue queue,
            size_t global_size,
            size_t local_size
            )
        :
            global_size_i(global_size),
            local_size_i(local_size)
    {
        this->device_i = device;
        this->queue_i = queue;
    }

    void setup() {
        bc::program zero_program = bc::program::build_with_source(
                zero_source,
                queue_i.get_context()
                );

        size_t global_size = global_size_i;
        size_t local_size = local_size_i;

        zero_f = [zero_program, global_size, local_size](
                bc::command_queue queue,
                size_t cl_offset,
                size_t size,
                bc::buffer buffer,
                bc::wait_list wait_list,
                Measurement::DataPoint& dp
                )
        {
            dp.set_name("zero");
            bc::kernel kernel = zero_program.create_kernel("zero");
            kernel.set_args(buffer, (cl_uint) (size / sizeof(cl_int)));
            bc::event event;
            event = queue.enqueue_1d_range_kernel(
                    kernel,
                    cl_offset / sizeof(cl_int),
                    global_size,
                    local_size,
                    wait_list
                    );
            dp.add_event() = event;
            return event;
        };
    }

    void teardown() {
        this->destroy_scheduler();
    }

    std::vector<std::tuple<size_t, uint64_t>> transfer_to_device_only(
            uint32_t repeat,
            std::vector<size_t> buffer_sizes
            ) {

        int ret = 0;
        uint32_t object_id = 0;
        std::vector<std::tuple<size_t, uint64_t>> transfer_time;

        long page_size = ::sysconf(_SC_PAGESIZE);
        assert(page_size != -1);

        size_t max_buffer_size = *std::max_element(
                buffer_sizes.begin(),
                buffer_sizes.end()
                );
        size_t data_size = ((max_buffer_size * repeat + page_size - 1)
                / page_size) * page_size;

        for (auto bs : buffer_sizes) {
            size_t transfer_size = bs * repeat;
            assert(transfer_size <= data_size);

            void *data = static_cast<int*>(::mmap(nullptr, data_size, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            assert(data != MAP_FAILED);

            // Pre-fault allocated pages
            volatile int throw_away = 0;
            int *idata = (int*)data;
            for (size_t i = 0; i < transfer_size / sizeof(int); ++i) {
                throw_away += idata[i];
            }

            std::future<std::deque<bc::event>> fevents;
            Measurement::Measurement measurement;

            object_id = new_scheduler(
                    bs,
                    data,
                    transfer_size
                    );

            ret = scheduler->enqueue(
                    zero_f,
                    object_id,
                    bs,
                    fevents,
                    measurement.add_datapoint()
                    );

            ret = scheduler->run();

            queue_i.finish();
            fevents.wait();
            auto events = fevents.get();

            auto times = measurement
                .get_execution_times_by_name<std::chrono::nanoseconds>(
                    std::regex("^BufferCache.*")
                    );

            for (auto& tuple : times) {
                std::string s;
                uint64_t t;
                std::tie (s, t) = tuple;

                transfer_time.emplace_back(bs, t);
            }

            destroy_scheduler();
            ::munmap(data, data_size);
        }

        return transfer_time;
    }

    void transfer_to_device_and_back() {
    }

private:
    uint32_t new_scheduler(size_t buffer_size, void *data_ptr, size_t data_size) {
        scheduler = std::make_shared<Clustering::SingleDeviceScheduler>();
        buffer_cache = std::make_shared<Clustering::SimpleBufferCache>(buffer_size);

        assert(true == scheduler->add_buffer_cache(buffer_cache));
        assert(true == scheduler->add_device(queue_i.get_context(), device_i));

        size_t pool_size = buffer_size * 2 + 1;
        assert(pool_size < device_i.global_memory_size());

        assert(true ==
                buffer_cache->add_device(
                    queue_i.get_context(),
                    device_i,
                    pool_size
                    ));

        uint32_t object_id = buffer_cache->add_object(
                data_ptr,
                data_size,
                Clustering::ObjectMode::ReadOnly
                );
        assert(object_id > 0);

        return object_id;
    }

    void destroy_scheduler() {
        scheduler.reset();
        buffer_cache.reset();
    }

    bc::device device_i;
    bc::command_queue queue_i;
    size_t global_size_i;
    size_t local_size_i;
    std::shared_ptr<Clustering::BufferCache> buffer_cache;
    std::shared_ptr<Clustering::DeviceScheduler> scheduler;
    Clustering::DeviceScheduler::FunUnary zero_f;
};

class Configuration {
public:
    Configuration()
    :
        global_size(1024),
        local_size(64),
        max_buffer_size(256ull << 20),
        repeat(10)
    {
        this->platform = bc::system::default_device().platform().id();
        this->device = bc::system::default_device().id();
    }

    int parse(int argc, char **argv) {
        char help_msg[] =
            "Usage: transfer_bench [OPTION]\n"
            "Options"
            ;

        po::options_description options(help_msg);
        options.add_options()
            ("help", "Produce help message")
            ("platform", "OpenCL Platform ID")
            ("device", "OpenCL Device ID")
            ("max-size", po::value<size_t>(), "Maximum transfer buffer size in Megabytes; Default: 256MB")
            ("repeat", po::value<uint32_t>(), "Number of transfers to make; Default: 10")
            // ("global_size", po::value<size_t>(), "Kernel Global Size; Default: 1024")
            // ("local_size", po::value<size_t>(), "Kernel Local Size; Default: 64")
            ;

        po::variables_map vm;
        po::store(
                po::command_line_parser(argc, argv)
                .options(options)
                .run(),
                vm
                );
        po::notify(vm);

        if (vm.count("help")) {
            std::cerr << options << std::endl;
            return -1;
        }

        if (vm.count("platform")) {
            this->platform = vm["platform"].as<cl_platform_id>();
        }

        if (vm.count("device")) {
            this->device = vm["device"].as<cl_device_id>();
        }

        if (vm.count("max-size")) {
            this->max_buffer_size = vm["max-size"].as<size_t>() << 20;
        }

        if (vm.count("repeat")) {
            this->repeat = vm["repeat"].as<uint32_t>();
        }

        if (vm.count("global_size")) {
            this->global_size = vm["global_size"].as<size_t>();
        }

        if (vm.count("local_size")) {
            this->local_size = vm["local_size"].as<size_t>();
        }

        return 1;
    }

    cl_platform_id platform;
    cl_device_id device;
    size_t global_size;
    size_t local_size;
    size_t max_buffer_size;
    uint32_t repeat;
};

int main(int argc, char **argv) {

    int rc = 0;

    Configuration config;
    rc = config.parse(argc, argv);
    if (rc == -1) {
        return 1;
    }

    std::vector<size_t> buffer_sizes;
    for (auto s = 1ull << 20; s < config.max_buffer_size; s = s * 2) {
        buffer_sizes.push_back(s);
    }
    buffer_sizes.push_back(config.max_buffer_size);

    bc::platform platform = bc::platform(config.platform);
    bc::device device;
    bool found_device = false;
    for (auto& d : platform.devices()) {
        if (d.id() == config.device) {
            device = d;
            found_device = true;
        }
    }
    if (not found_device) {
        std::cerr
            << "Error: Could not find device with ID "
            << config.device
            << std::endl
            ;
        return 1;
    }
    bc::context context = bc::context(device);
    bc::command_queue queue = bc::command_queue(
            context,
            device,
            bc::command_queue::enable_profiling
            );

    std::cout
        << "Running TransferBench for "
        << "device " << device.name()
        << " on platform " << device.platform().name()
        << std::endl
        ;

    TransferBench tb(
            device,
            queue,
            config.global_size,
            config.local_size
            );
    tb.setup();
    auto transfer_time = tb.transfer_to_device_only(config.repeat, buffer_sizes);
    tb.teardown();

    std::cout
        << "Buffer_Size_(bytes)\ttransfer_to_device_(ns)"
        << std::endl
        ;

    for (auto& tuple : transfer_time) {
        size_t size;
        uint64_t nanos;
        std::tie (size, nanos) = tuple;

        std::cout
            << std::setw(10)
            << size
            << '\t'
            << nanos
            << '\n'
            ;
    }
    std::cout << std::flush;


    return 0;
}
