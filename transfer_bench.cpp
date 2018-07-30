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
#include <sys/syscall.h>
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

        uint32_t object_id = 0;
        std::vector<std::tuple<size_t, uint64_t>> transfer_time;

        long page_size = ::sysconf(_SC_PAGESIZE);
        assert(page_size != -1);
        assert(DATA_SIZE % page_size == 0);

        // Calculate size biggest array we will use and round up to page_size
        size_t max_required_size = ((*std::max_element(
                buffer_sizes.begin(),
                buffer_sizes.end()
                ) * repeat + page_size - 1) / page_size) * page_size;

        // Reserve enough address space max_required_size,
        // but don't actually allocate pages (relying on kernel to dedupliate pages)
        void *reserved = ::mmap(nullptr, max_required_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        for (auto bs : buffer_sizes) {
            size_t transfer_size = bs * repeat;

            // Create in-memory file and map it into address space
            //int fdesc = ::memfd_create("DATA", 0); // Requires glibc >= 2.27
	    int fdesc = ::syscall(SYS_memfd_create, "DATA", 0);
            assert(fdesc != -1);
            assert(0 == ftruncate(fdesc, DATA_SIZE));
            void *data = ::mmap(reserved, DATA_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fdesc, 0);
            assert(data != MAP_FAILED);
            assert(data == reserved);

            // Pre-fault pages and ensure they are allocated
            int *idata = static_cast<int*>(data);
            for (size_t i = 0; i < DATA_SIZE / sizeof(int); ++i) {
                idata[i] = i;
            }

            // Clone the file in the address space until the array is big enough
            std::vector<void*> maps = {data};
            for (
                    auto mapped_size = DATA_SIZE;
                    mapped_size < transfer_size;
                    mapped_size += DATA_SIZE
                )
            {
                char *cdata = static_cast<char*>(data);
                char *location = cdata + mapped_size;
                void *clone = ::mmap(location, DATA_SIZE, PROT_READ, MAP_PRIVATE | MAP_FIXED, fdesc, 0);
                assert(clone != MAP_FAILED);
                assert(location == clone);
                maps.push_back(clone);
            }

            // Pre-fault pages
            volatile int throw_away = 0;
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

            assert(true ==
                    scheduler->enqueue(
                        zero_f,
                        object_id,
                        bs,
                        fevents,
                        measurement.add_datapoint()
                        ));

            assert(true ==
                    scheduler->run());

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

            for (void *map : maps) {
                assert(0 == ::munmap(map, DATA_SIZE));
            }
            assert(0 == ::close(fdesc));
        }

        assert(0 == ::munmap(reserved, max_required_size));

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

    size_t DATA_SIZE = 128ull << 20; // 128 MB, much more than CPU's LLC
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
        this->platform = 0;
        this->device = 0;
    }

    int parse(int argc, char **argv) {
        char help_msg[] =
            "Usage: transfer_bench [OPTION]\n"
            "Options"
            ;

        po::options_description options(help_msg);
        options.add_options()
            ("help", "Produce help message")
            ("platform", po::value<uint32_t>(), "OpenCL Platform ID")
            ("device", po::value<uint32_t>(), "OpenCL Device ID")
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
            this->platform = vm["platform"].as<uint32_t>();
        }

        if (vm.count("device")) {
            this->device = vm["device"].as<uint32_t>();
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

    uint32_t platform;
    uint32_t device;
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

    bc::platform platform = bc::system::platforms()[config.platform];
    bc::device device = platform.devices()[config.device];
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
