#ifndef CL_INITIALIZER_HPP
#define CL_INITIALIZER_HPP

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {
    class CLInitializer {
    public:
        CLInitializer();
        ~CLInitializer();

        int choose_platform_interactive();
        int choose_device_interactive();

        cl::Platform& get_platform();
        cl::Context& get_context();
        cl::Device& get_device();
        cl::CommandQueue& get_commandqueue();

    private:
        cl::Platform *platform_;
        cl::Context *context_;
        cl::Device *device_;
        cl::CommandQueue *queue_;
    };
}

#endif /* CL_INITIALIZER_HPP */
