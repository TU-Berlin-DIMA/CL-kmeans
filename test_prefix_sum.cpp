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
       std::cerr << "Platform size 0" << std::endl;
       return -1;
    }

    // Show platforms
    err = cle::show_platforms(platforms);
    if (err != CL_SUCCESS) {
        return err;
    }


    // Select the default platform and create a context
    // using this platform and the GPU
    cl_context_properties cps[3] = { 
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[0])(), 0
    };
    cl::Context context(CL_DEVICE_TYPE_ALL, cps, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Context setup failed with: "
            << cle::opencl_error_string(err)
            << std::endl;
        return err;
    }

    // Show context info
    std::vector<cl::Device> context_devices;
    std::vector<cl_context_properties> context_properties;
    std::string device_name;

    cle_sanitize_val_return(
            context.getInfo(CL_CONTEXT_DEVICES, &context_devices));

    std::cout << "Selected devices:" << std::endl;
    for (auto device : context_devices) {
        cle_sanitize_val_return(
                device.getInfo(CL_DEVICE_NAME, &device_name));
        std::cout << "  - " << device_name << std::endl;
    }

    cle_sanitize_val_return(
            context.getInfo(CL_CONTEXT_PROPERTIES, &context_properties));

    // Get a list of devices on this platform
    std::vector<cl::Device> devices;
    cle_sanitize_val_return(
            context.getInfo(CL_CONTEXT_DEVICES, &devices));

    // Create a command queue and use the first device
    cl::CommandQueue queue;
    cle_sanitize_ref_return(
            queue = cl::CommandQueue(context, devices[0], 0, &err),
            err
            );

    // Create kernel
    std::function<cle::Prefix_Sum_Kernel::kernel_functor::type_>
        prefix_sum_kernel
        = cle::Prefix_Sum_Kernel::get_kernel(context, err);
    if (err != CL_SUCCESS) {
        return err;
    }

    cl::Buffer idata_buffer(context, CL_MEM_READ_WRITE, idata.size() * sizeof(decltype(idata)::value_type));
    cl::Buffer odata_buffer(context, CL_MEM_READ_WRITE, odata.size() * sizeof(decltype(odata)::value_type));
    cl::Buffer carry_buffer(context, CL_MEM_READ_WRITE, sizeof(decltype(carry)));

    cle_sanitize_val_return(
            queue.enqueueWriteBuffer(
                idata_buffer, 
                CL_FALSE, 
                0,
                idata.size() * sizeof(decltype(idata)::value_type),
                idata.data()
                ));

    cle_sanitize_val_return(
            queue.enqueueWriteBuffer(
                carry_buffer,
                CL_FALSE,
                0,
                sizeof(decltype(carry)),
                &carry
                ));

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
    cle_sanitize_val_return(
        queue.enqueueReadBuffer(
            odata_buffer,
            CL_TRUE,
            0,
            odata.size() * sizeof(decltype(odata)::value_type),
            odata.data()
            ));

    // Get carry buffer
    cle_sanitize_val_return(
        queue.enqueueReadBuffer(
            carry_buffer,
            CL_TRUE,
            0,
            sizeof(decltype(carry)), &carry
            ));

    // Print output
    std::cout << "CL output" << std::endl;
    print_vector(odata);
    std::cout << "CL prefix carry: " << carry << std::endl;

}
