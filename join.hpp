/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */


#ifndef JOIN_HPP_
#define JOIN_HPP_

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {
  
  /*
   * cl_loop_join
   * 
   * Performs join on table_A and table_B
   *
   * Input: 2 buffers with cl_uint arrays
   * Output: Buffer pointer with Pair array 
   *
   * Invariants:
   */
  void cl_loop_join(cl::Buffer& table_A, cl::Buffer& table_B, cl::Buffer* output);


}

#endif /* JOIN_HPP_ */
