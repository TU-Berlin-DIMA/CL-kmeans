
#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include <string>
#include <fstream>
#include <utility>
#include <iostream>
#include <vector>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {
    cl::Program make_program(cl::Context context, std::string file) {
        cl_int err = CL_SUCCESS;
        
        std::ifstream sourceFile(file);
        if (not sourceFile.good()) {
            std::cerr << "Failed to open program file " << file << std::endl;
        }
        
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        cl::Program program = cl::Program(context, source, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create program" << std::endl;
        }

        std::vector<cl::Device> context_devices;
        context.getInfo(CL_CONTEXT_DEVICES, &context_devices);

        err = program.build(context_devices);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to build program" << std::endl;
        }

        return program;
    }
}


#endif /* HELPERS_HPP_ */
