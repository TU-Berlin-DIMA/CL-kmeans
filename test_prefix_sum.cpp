#include <iostream>
#include <vector>
#include <cstdint>
#include <functional>
#include <algorithm>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "prefix_sum.hpp"

void print_vector(std::vector<cl_uint> const& vec) {
    for_each(vec.begin(), vec.end(), [](cl_uint x){ std::cout << x << " "; });
    std::cout << std::endl;
}

int main() {
    const uint32_t DATA_SIZE = 8;
    const uint32_t GLOBAL_SIZE = DATA_SIZE / cle::Prefix_Sum_Kernel::WORK_ITEM_SIZE;
    const uint32_t LOCAL_SIZE = GLOBAL_SIZE;

    std::vector<cl_uint> idata(DATA_SIZE), odata(DATA_SIZE);
    cl_uint carry(0);

    cl_int err = CL_SUCCESS;

    // Initialize input data
    for (uint32_t i = 0; i != idata.size(); ++i) {
        idata[i] = i + 1;
    }
    std::cout << "CL input" << std::endl;
    print_vector(idata);

    // Get available platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
       std::cout << "Platform size 0\n";
       return -1;
    }

    // Show platforms
    std::string platform_name, device_name, device_version;
    cl_device_type device_type;
    std::vector<cl::Device> devices_list;
    std::cout << "Platforms:" << std::endl;
    for (auto platform : platforms) {
        platform.getInfo(CL_PLATFORM_NAME, &platform_name);
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_list);

        std::cout << "  - " << platform_name << std::endl;
        for (auto device : devices_list) {
            device.getInfo(CL_DEVICE_TYPE, &device_type);
            device.getInfo(CL_DEVICE_NAME, &device_name);
            device.getInfo(CL_DEVICE_VERSION, &device_version);

            std::cout << "    + Device: ";
            while (device_type != 0) {
                switch(device_type) {
                    case CL_DEVICE_TYPE_CPU:
                        std::cout << "CPU ";
                        device_type ^= CL_DEVICE_TYPE_CPU;
                        break;
                    case CL_DEVICE_TYPE_GPU:
                        std::cout << "GPU ";
                        device_type ^= CL_DEVICE_TYPE_GPU;
                        break;
                    case CL_DEVICE_TYPE_ACCELERATOR:
                        std::cout << "ACCELERATOR ";
                        device_type ^= CL_DEVICE_TYPE_ACCELERATOR;
                        break;
                    case CL_DEVICE_TYPE_DEFAULT:
                        std::cout << "default device type ";
                        device_type ^= CL_DEVICE_TYPE_DEFAULT;
                        break;
                }
            }
            std::cout << device_name << ", " << device_version << std::endl;
        }
    }

    // Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[3] = { 
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[0])(), 0
    };
    cl::Context context(CL_DEVICE_TYPE_ALL, cps, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Context setup failed with: ";
        switch (err) {
            case CL_INVALID_PROPERTY:
                std::cerr << "invalid property";
                break;
            case CL_INVALID_VALUE:
                std::cerr << "invalid value";
                break;
            case CL_INVALID_DEVICE:
                std::cerr << "invalid device";
                break;
            case CL_DEVICE_NOT_AVAILABLE:
                std::cerr << "device not available";
                break;
            case CL_OUT_OF_HOST_MEMORY:
                std::cerr << "out of host memory";
                break;
        }
        std::cerr << std::endl;
    }

    // Show context info
    std::vector<cl::Device> context_devices;
    std::vector<cl_context_properties> context_properties;
    err = context.getInfo(CL_CONTEXT_DEVICES, &context_devices);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to get context device info" << std::endl;
    }
    else {
        std::cout << "Selected devices:" << std::endl;
        for (auto device : context_devices) {
            device.getInfo(CL_DEVICE_NAME, &device_name);
            std::cout << "  - " << device_name << std::endl;
        }
    }
    err = context.getInfo(CL_CONTEXT_PROPERTIES, &context_properties);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to get context platform info" << std::endl;
    }

    // Get a list of devices on this platform
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    // Create a command queue and use the first device
    cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

    // Create kernel
    std::function<cle::Prefix_Sum_Kernel::kernel_functor::type_> prefix_sum_kernel = cle::Prefix_Sum_Kernel::get_kernel(context);

    cl::Buffer idata_buffer(context, CL_MEM_READ_WRITE, idata.size() * sizeof(decltype(idata)::value_type));
    cl::Buffer odata_buffer(context, CL_MEM_READ_WRITE, odata.size() * sizeof(decltype(odata)::value_type));
    cl::Buffer carry_buffer(context, CL_MEM_READ_WRITE, sizeof(decltype(carry)));

    err = queue.enqueueWriteBuffer(idata_buffer, CL_FALSE, 0, idata.size() * sizeof(decltype(idata)::value_type), idata.data());
    if (err != CL_SUCCESS) {
        std::cerr << "enqueueWriteBuffer failed" << std::endl;
        return -1;
    }

    queue.enqueueWriteBuffer(carry_buffer, CL_FALSE, 0, sizeof(decltype(carry)), &carry);
    if (err != CL_SUCCESS) {
       std::cerr << "enqueueWriteBuffer failed" << std::endl;
       return -1;
     }

    // Invoke kernel
    prefix_sum_kernel(
            cl::EnqueueArgs(queue, cl::NDRange(GLOBAL_SIZE), cl::NDRange(LOCAL_SIZE)),
            idata_buffer,
            odata_buffer,
            carry_buffer,
            cl::Local(sizeof(decltype(idata)::value_type) * idata.size()),
            idata.size()
            );

    // Get output buffer
    err = queue.enqueueReadBuffer(odata_buffer, CL_TRUE, 0, odata.size() * sizeof(decltype(odata)::value_type), odata.data());
    if (err != CL_SUCCESS) {
        std::cerr << "enqueueReadBuffer failed" << std::endl;
        return -1;
    }

    // Get carry buffer
    err = queue.enqueueReadBuffer(carry_buffer, CL_TRUE, 0, sizeof(decltype(carry)), &carry);
    if (err != CL_SUCCESS) {
        std::cerr << "enqueueReadBuffer failed" << std::endl;
        return -1;
    }

    // Print output
    std::cout << "CL output" << std::endl;
    print_vector(odata);
    std::cout << "CL prefix carry: " << carry << std::endl;

}
