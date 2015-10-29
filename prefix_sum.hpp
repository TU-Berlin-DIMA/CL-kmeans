
#ifndef PREFIX_SUM_HPP_
#define PREFIX_SUM_HPP_

#include <function>
#include <string>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "helpers.hpp"

namespace cle {
  
  typedef cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::cl_uint&> Prefix_Sum_Kernel;

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
  std::function<Prefix_Sum_Kernel::type_> cl_prefix_sum_kernel(cl::Context& context) {
    cl::Program program = make_program(context, "cl_prefix_sum.cl");
    return Prefix_Sum_Kernel(program, "cl_prefix_sum");
  }



}

#endif /* PREFIX_SUM_HPP_ */
