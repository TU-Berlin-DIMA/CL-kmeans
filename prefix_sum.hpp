
#ifndef PREFIX_SUM_HPP_
#define PREFIX_SUM_HPP_

#include <functional>
#include <string>
#include <cstdint>
#include <string>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "helpers.hpp"

namespace cle {
  
    class Prefix_Sum_Kernel {
    public:
        static constexpr const char* PROGRAM_FILE = "prefix_sum.cl";
        static constexpr const char* KERNEL_NAME = "prefix_sum";
        static constexpr uint32_t WORK_ITEM_SIZE = 2;

        typedef cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg, cl_ulong> kernel_functor;

        /*
        * cl_prefix_sum
        * 
        * Performs exclusive prefix sum on input
        *
        * Input: Buffer with cl_uint array
        * Output: Buffer with cl_uint array 
        *         cl_uint with total sum
        *
        * Invariants:
        *  - input.size() == output.size()
        */
        static std::function<kernel_functor::type_> get_kernel(cl::Context& context) {
            cl::Program program = make_program(context, PROGRAM_FILE);

            cl_int err = CL_SUCCESS;
            auto kf = kernel_functor(program, KERNEL_NAME, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to make kernel with: ";
                switch(err) {
                    case CL_INVALID_PROGRAM:
                        std::cerr << "invalid program" << std::endl;
                        break;
                    case CL_INVALID_PROGRAM_EXECUTABLE:
                        std::cerr << "invalid program executable" << std::endl;
                        break;
                    case CL_INVALID_KERNEL_NAME:
                        std::cerr << "invalid kernel name" << std::endl;
                        break;
                    case CL_INVALID_KERNEL_DEFINITION:
                        std::cerr << "invalid kernel definition" << std::endl;
                        break;
                    case CL_INVALID_VALUE:
                        std::cerr << "invalid value" << std::endl;
                        break;
                    case CL_OUT_OF_RESOURCES:
                        std::cerr << "out of resources" << std::endl;
                        break;
                    case CL_OUT_OF_HOST_MEMORY:
                        std::cerr << "out of host memory" << std::endl;
                        break;
                }

                if (err == CL_INVALID_PROGRAM || err == CL_INVALID_PROGRAM_EXECUTABLE) {
                    // Print build logs
                    std::cout << "Printing program build log(s) for further diagnosis:" << std::endl;

                    std::string build_log;
                    std::vector<cl::Device> context_devices;

                    context.getInfo(CL_CONTEXT_DEVICES, &context_devices);
                    for (cl::Device device : context_devices) {
                        program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log);
                        std::cout << build_log << std::endl;
                    }
                }
            }

            return kf;
        }
    };


}

#endif /* PREFIX_SUM_HPP_ */
