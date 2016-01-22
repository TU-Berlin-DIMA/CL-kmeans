#include "cl_initializer.hpp"

cle::CLInitializer::CLInitializer() :
    platform_(NULL),
    context_(NULL),
    device_(NULL),
    queue_(NULL)
{}

cle::CLInitializer::~CLInitializer() {
    if (this->context_ != NULL) {
        delete this->context_;
    }

    if (this->device_ != NULL) {
        delete this->device_;
    }

    if (this->platform_ != NULL) {
        delete this->platform_;
    }

    if (this->queue_ != NULL) {
        delete this->queue_;
    }
}


int cle::CLInitializer::choose_platform_interactive() {
    cl_int err;
    unsigned int choice;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        std::cerr << "Error: no platforms found" << std::endl;
        return -1;
    }

    // Display platform list
    err = show_platforms(platforms);
    if (err != CL_SUCCESS) {
        return -1;
    }

    // Ask user for selection
    std::cout << "Enter platform number: " << std::flush;
    std::cin >> choice;
    std::cout << std::endl;
    this->platform_ = new cl::Platform(platforms[choice]);

    // Create context
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[choice])(), 0
    };

    cle_sanitize_ref_return(
            this->context_ = new cl::Context(
                CL_DEVICE_TYPE_ALL,
                cps,
                NULL,
                NULL,
                &err
                ), err);

    return 1;
}

int cle::CLInitializer::choose_device_interactive() {
    cl_int err;
    std::vector<cl::Device> devices;

    cle_sanitize_val_return(
            this->context_->getInfo(
                CL_CONTEXT_DEVICES,
                &devices
                ));

    // TODO: show devices, choose device
    this->device_ = new cl::Device(devices[0]);

    // Create command queue for device
    cle_sanitize_ref_return(
            this->queue_ = new cl::CommandQueue(
                *this->context_,
                *this->device_,
                CL_QUEUE_PROFILING_ENABLE,
                &err
                ), err);

    return 1;
}

cl::Platform& cle::CLInitializer::get_platform() {
    return *this->platform_;
}

cl::Context& cle::CLInitializer::get_context() {
    return *this->context_;
}

cl::Device& cle::CLInitializer::get_device() {
    return *this->device_;
}

cl::CommandQueue& cle::CLInitializer::get_commandqueue() {
    return *this->queue_;
}
