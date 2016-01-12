
#ifndef PREFIX_SUM_HPP_
#define PREFIX_SUM_HPP_

#include <functional>
#include <string>
#include <cstdint>
#include <string>
#include <cassert>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "helpers.hpp"

namespace cle {
  
    class Prefix_Sum_Kernel {
    public:
        static constexpr uint32_t WORK_ITEM_SIZE = 2;

        cl_int initialize(cl::Context& context) {
            cl_int error_code = CL_SUCCESS;

            cl::Program program = make_program(context, PROGRAM_FILE, error_code);
            if (error_code != CL_SUCCESS) {
                return error_code;
            }

            kernel_functor_ = Base_Kernel(program,KERNEL_NAME, &error_code);
            sanitize_make_kernel(error_code, context, program);

            return error_code;
        }

        /*
        * kernel prefix_sum
        * 
        * Performs exclusive prefix sum on input
        *
        * Input
        *       Buffer with cl_uint array
        *
        * Output
        *       Buffer with cl_uint array 
        *       cl_uint with total sum
        *
        */
        void operator() (
                cl::EnqueueArgs const& args,
                TypedBuffer<cl_uint>& in,
                TypedBuffer<cl_uint>& out_sums,
                TypedBuffer<cl_uint>& out_carrys) {

            assert(out_sums.size() >= in.size());
            assert(out_carrys.size() >= 1);

            assert(in.size() >= 2);
            assert(in.size() % 2 == 0);


            size_t bytes = sizeof(cl_uint) * in.size();
            cl::LocalSpaceArg local_buf = cl::Local(bytes);

            kernel_functor_(args, in, out_sums, out_carrys,
                    local_buf, bytes);
        }

    private:
        typedef cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg, cl_ulong> Base_Kernel;
        typedef std::function<Base_Kernel::type_> Kernel_Functor;

        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("prefix_sum.cl");
        static constexpr const char* KERNEL_NAME = "prefix_sum";

        Kernel_Functor kernel_functor_;
    };
}

#endif /* PREFIX_SUM_HPP_ */
