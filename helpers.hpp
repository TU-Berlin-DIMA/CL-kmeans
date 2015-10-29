
#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include <string>
#include <fstream>
#include <utility>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {
  cl::Program make_program(cl::Context context, std::string file) {
    std::ifstream sourceFile(file);
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources cl::source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    cl::Program program = cl::Program(context, source);

    return program;
  }
}


#endif /* HELPERS_HPP_ */
