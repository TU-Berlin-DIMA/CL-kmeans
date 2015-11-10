
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
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("prefix_sum.cl");
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
            auto kf = kernel_functor(program,KERNEL_NAME, &err);
            sanitize_make_kernel(err, context, program);

            return kf;
        }
    };


}

#endif /* PREFIX_SUM_HPP_ */
