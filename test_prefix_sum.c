#define PROGRAM_FILE "prefix_sum.cl"
#define KERNEL_FUNC "prefix_sum"
#define DATA_SIZE 8
#define WORK_ITEM_SIZE 2
#define LOCAL_WORK_SIZE (DATA_SIZE / WORK_ITEM_SIZE)

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename, const char* options) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   uint64_t program_size, log_size = 0;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   err = 0;
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program */
   err = 0;
   err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

uint32_t prefix_sum(const uint32_t *in, uint32_t *out, const uint64_t size) {
  out[0] = 0;
  for (uint64_t i = 1; i != size; ++i) {
    out[i] = in[i - 1] + out[i - 1];
  }

  return in[size - 1] + out[size - 1];
}

void print_buffer(const uint32_t *buf, const uint64_t size) {
  for (uint64_t i = 0; i != size; ++i) {
    printf("%u ", buf[i]);
  }
  printf("\n");
}

int main() {

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program p_prefix_sum;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int err;
   uint64_t local_size, global_size;

   /* Data and buffers */
   const uint64_t data_size = DATA_SIZE;
   uint32_t idata[DATA_SIZE];
   uint32_t odata[DATA_SIZE];
   uint32_t check_data[DATA_SIZE];
   cl_mem idata_buffer, odata_buffer, carry_buffer;
   cl_int num_groups;

   /* Initialize data */
   for (uint64_t i = 0; i != DATA_SIZE; ++i) {
     /*idata[i] = rand();*/
     idata[i] = i + 1;
   }

   /* Create device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build programs */
   p_prefix_sum = build_program(context, device, PROGRAM_FILE, "");

   /* Create table buffers */
   global_size = DATA_SIZE / WORK_ITEM_SIZE;
   local_size = LOCAL_WORK_SIZE;
   num_groups = global_size / local_size;
   idata_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(uint32_t), idata, &err);
   odata_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_ALLOC_HOST_PTR, DATA_SIZE * sizeof(uint32_t), NULL, &err);
   carry_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
       CL_MEM_ALLOC_HOST_PTR, global_size * sizeof(uint32_t), NULL, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Create a kernel */
   kernel = clCreateKernel(p_prefix_sum, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &idata_buffer);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &odata_buffer);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &carry_buffer);
   err |= clSetKernelArg(kernel, 3, DATA_SIZE * sizeof(cl_uint), NULL);
   err |= clSetKernelArg(kernel, 4, sizeof(uint64_t), &data_size);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   /* Enqueue prefix sum kernel */
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, odata_buffer, CL_TRUE, 0, 
         DATA_SIZE * sizeof(cl_uint), odata, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   /* Check result */
   printf("Checking result\n");
   memset(check_data, 0, DATA_SIZE);
   prefix_sum(idata, check_data, DATA_SIZE);
   err = memcmp(odata, check_data, DATA_SIZE);
   if (err != 0) {
     printf("Result incorrect\n");
   }
   else {
     printf("Result correct\n");
   }

   printf("idata:\n");
   print_buffer(idata, DATA_SIZE);
   printf("check data:\n");
   print_buffer(check_data, DATA_SIZE);
   printf("odata:\n");
   print_buffer(odata, DATA_SIZE);

   /* Deallocate resources */
   clReleaseKernel(kernel);
   clReleaseMemObject(idata_buffer);
   clReleaseMemObject(odata_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(p_prefix_sum);
   clReleaseContext(context);

   return 0;
}
