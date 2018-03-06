/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef REDUCE_VECTOR_PARCOL_API_HPP
#define REDUCE_VECTOR_PARCOL_API_HPP

#include "kernel_path.hpp"

#include <clext.hpp>

#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

template <typename CL_TYPE, typename CL_INT> class ReduceVectorParcolAPI {
public:
  cl_int initialize(cl::Context &context) {
    cl_int error_code = CL_SUCCESS;

    std::string defines;
    if (std::is_same<cl_uint, CL_INT>::value) {
      defines = "-DTYPE32";
    } else if (std::is_same<cl_ulong, CL_INT>::value) {
      defines = "-DTYPE64";
    } else {
      assert(false);
    }
    defines += " -DCL_TYPE=";
    defines += cle::OpenCLType::to_str<CL_TYPE>();

    defines += " -DMAX_WORKGROUP_SIZE=";
    defines += std::to_string(MAX_WORKGROUP_SIZE);

    cl::Program program =
        make_program(context, PROGRAM_FILE, defines, error_code);
    if (error_code != CL_SUCCESS) {
      return error_code;
    }

    reduce_vector_parcol_compact_kernel_.reset(
        new cl::Kernel(program, COMPACT_KERNEL_NAME, &error_code));
    sanitize_make_kernel(error_code, context, program);

    if (error_code != CL_SUCCESS) {
        return error_code;
    }

    reduce_vector_parcol_inner_kernel_.reset(
        new cl::Kernel(program, INNER_KERNEL_NAME, &error_code));
    sanitize_make_kernel(error_code, context, program);

    return error_code;
  }

  /*
  * kernel reduce vector parcol
  */
  cl_int operator()(cl::EnqueueArgs const &args, CL_INT num_cols,
                    CL_INT num_rows, TypedBuffer<CL_TYPE> &data,
                    cl::Event &event) {

    assert(data.size() >= num_cols * num_rows);

    uint64_t global_size = data.size() / 2;
    uint32_t round = 0;
    CL_INT data_size = data.size();
    while (data_size > num_rows && data_size > 2 * MAX_WORKGROUP_SIZE) {
      assert(global_size * 2 == data_size);
      assert(global_size % num_rows == 0);

      cle_sanitize_val_return(
          reduce_vector_parcol_compact_kernel_->setArg(0, (cl::Buffer &)data));

      cle_sanitize_val_return(
          reduce_vector_parcol_compact_kernel_->setArg(1, data_size));

      cle_sanitize_val_return(
              args.queue_.enqueueNDRangeKernel(
                  *reduce_vector_parcol_compact_kernel_,
                  args.offset_,
                  cl::NDRange(global_size),
                  cl::NullRange,
                  &args.events_, &event));

      global_size /= 2;
      data_size = global_size * 2;
      ++round;
    }

    if (data_size != num_rows && data_size == 2 * MAX_WORKGROUP_SIZE) {

      cle_sanitize_val_return(
              reduce_vector_parcol_inner_kernel_->setArg(0, (cl::Buffer &)data));

      cle_sanitize_val_return(
              reduce_vector_parcol_inner_kernel_->setArg(1, num_cols));

      cle_sanitize_val_return(
              reduce_vector_parcol_inner_kernel_->setArg(2, num_rows));

      cle_sanitize_val_return(
              args.queue_.enqueueNDRangeKernel(
                  *reduce_vector_parcol_inner_kernel_,
                  args.offset_,
                  cl::NDRange(MAX_WORKGROUP_SIZE),
                  cl::NDRange(MAX_WORKGROUP_SIZE),
                  &args.events_, &event));
    }

    return CL_SUCCESS;
  }

private:
  static constexpr uint32_t MAX_WORKGROUP_SIZE = 64;

  static constexpr const char *PROGRAM_FILE =
      CL_KERNEL_FILE_PATH("reduce_vector_parcol.cl");
  static constexpr const char *COMPACT_KERNEL_NAME = "reduce_vector_parcol_compact";
  static constexpr const char *INNER_KERNEL_NAME = "reduce_vector_parcol_inner";

  std::shared_ptr<cl::Kernel> reduce_vector_parcol_compact_kernel_;
  std::shared_ptr<cl::Kernel> reduce_vector_parcol_inner_kernel_;
};
}

#endif /* REDUCE_VECTOR_PARCOL_API_HPP */
