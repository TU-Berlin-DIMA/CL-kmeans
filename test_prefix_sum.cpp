#include <iostream>
#include <vector>
#include <cstdint>

int main() {
  const uint32_t DATA_SIZE = 8;
  const uint32_t WORK_ITEM_SIZE = 2;
  const uint32_t GLOBAL_SIZE = DATA_SIZE / WORK_ITEM_SIZE;
  const uint32_t LOCAL_SIZE = GLOBAL_SIZE;

  std::vector<cl_uint> idata(DATA_SIZE), odata(DATA_SIZE);
  cl_uint carry(0);


  cl_int err = CL_SUCCESS;

  // Initialize input data
  for (uint32_t i = 0; i != idata.size(); ++i) {
    idata[i] = i + 1;
  }

  // Get available platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    std::cout << "Platform size 0\n";
    return -1;
  }

  // Select the default platform and create a context using this platform and the GPU
  cl_context_properties cps[3] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)(platforms[0])(), 
    0 
  };
  cl::Context context(CL_DEVICE_TYPE_GPU, cps);

  // Get a list of devices on this platform
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

  // Create a command queue and use the first device
  std::CommandQueue queue = CommandQueue(context, devices[0]);

  // Create kernel
  std::function<Prefix_Sum_Kernel::type_> prefix_sum_kernel = cle::cl_prefix_sum_kernel(context);

  cl::Buffer idata_buffer(context, CL_MEM_READ_WRITE, idata.size() * sizeof(idata.value_type));
  cl::Buffer odata_buffer(context, CL_MEM_READ_WRITE, odata.size() * sizeof(odata.value_type));

  queue.enqueueWriteBuffer(idata_buffer, CL_TRUE, 0, idata.size() * sizeof(idata.value_type), idata.data());

  // Invoke kernel
  prefix_sum_kernel(cl::EnqueueArgs(queue, cl::NDRange(GLOBAL_SIZE), cl::NDRange(LOCAL_SIZE)), idata_buffer, odata_buffer, carry);

  queue.enqueueReadBuffer(odata_buffer, CL_TRUE, 0, odata.size() * sizeof(odata.value_type), odata.data());

}
