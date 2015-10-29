#define PROGRAM_FILE "loop_join.cl"
#define KERNEL_FUNC "loop_join"
#define PREFIX_SUM_FILE "prefix_sum.cl"
#define PREFIX_SUM_FUNC "prefix_sum"
#define TABLE_A_SIZE 64
#define TABLE_B_SIZE 8
#define LOCAL_WORK_SIZE 4
#define WORK_ITEM_SIZE 4

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "pair.h"

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
   size_t program_size, log_size;
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
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program */
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

int main() {

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program p_loop_join_count, p_loop_join, p_prefix_sum;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int i, j, err;
   size_t local_size, global_size;
   cl_int num_groups;

   global_size = TABLE_A_SIZE / WORK_ITEM_SIZE;
   local_size = LOCAL_WORK_SIZE;
   num_groups = global_size / local_size;

   /* Data and buffers */
   const unsigned long long table_A_size = TABLE_A_SIZE;
   const unsigned long long table_B_size = TABLE_B_SIZE;
   int table_A[TABLE_A_SIZE];
   int table_B[TABLE_B_SIZE];
   pair *table_joined = NULL;
   unsigned int *table_joined_partition_sizes = NULL; 
   unsigned int *table_joined_partition_offsets = NULL;
   unsigned int table_joined_size = 0;
   unsigned int carry[num_groups];
   cl_mem table_A_buffer, table_B_buffer, table_joined_buffer, table_joined_partition_sizes_buffer, table_joined_partition_offsets_buffer, carry_buffer;

   /* Initialize tables */
   printf("Table A:\n");
   for (i = 0; i != TABLE_A_SIZE; ++i) {
     table_A[i] = rand() % (TABLE_B_SIZE + 1);
     printf("%d ", table_A[i]);
   }

   printf("\nTable B: \n");
   j = 0;
   for (i = 0; i != TABLE_B_SIZE; ++i) {
     table_B[i] = j;

     if (i == TABLE_B_SIZE / 2)
       j = 0;
     else
       ++j;

     printf("%d ", table_B[i]);
   }
   printf("\n");

   /* Create device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build programs */
   p_loop_join_count = build_program(context, device, PROGRAM_FILE, "-D COUNT_RESULT_ROWS");
   p_loop_join = build_program(context, device, PROGRAM_FILE, "");
   p_prefix_sum = build_program(context, device, PREFIX_SUM_FILE, "");

   /* Create table buffers */
   table_A_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, TABLE_A_SIZE * sizeof(int), table_A, &err);
   table_B_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, TABLE_B_SIZE * sizeof(int), table_B, &err);
   table_joined_partition_sizes_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_ALLOC_HOST_PTR, global_size * sizeof(cl_uint), NULL, &err);
   table_joined_partition_offsets_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
       CL_MEM_ALLOC_HOST_PTR, global_size * sizeof(cl_uint), NULL, &err);
   carry_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
       CL_MEM_ALLOC_HOST_PTR, num_groups * sizeof(cl_uint), NULL, &err);
 if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   table_joined_partition_sizes = malloc(global_size * sizeof(unsigned long long));
   if (table_joined_partition_sizes == NULL) {
     perror("Couldn't allocate memory");
     exit(1);
   }

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Create a kernel */
   kernel = clCreateKernel(p_loop_join_count, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   }

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &table_A_buffer);
   err |= clSetKernelArg(kernel, 1, sizeof(unsigned long long), &table_A_size);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &table_B_buffer);
   err |= clSetKernelArg(kernel, 3, sizeof(unsigned long long), &table_B_size);
   err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &table_joined_partition_sizes_buffer);
   err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), NULL);
   err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), NULL);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   /* Enqueue loop join count kernel */
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, table_joined_partition_sizes_buffer, CL_TRUE, 0, 
         global_size * sizeof(cl_uint), table_joined_partition_sizes, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   /* Check result */
   printf("Partition sizes of join table: ");
   for (i = 0; i != global_size; ++i) {
     printf("%u ", table_joined_partition_sizes[i]);
   }
   printf("\n");

   /* Create prefix-sum kernel */
   kernel = clCreateKernel(p_prefix_sum, PREFIX_SUM_FUNC, &err);
   if (err < 0) {
     perror("Couldn't create prefix_sum kernel");
     exit(1);
   }

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &table_joined_partition_sizes_buffer);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &table_joined_partition_offsets_buffer);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &carry_buffer);
   err |= clSetKernelArg(kernel, 3, global_size * sizeof(cl_uint), NULL);
   err |= clSetKernelArg(kernel, 4, sizeof(uint64_t), &global_size);
   if(err < 0) {
     perror("Couldn't create a kernel argument");
     exit(1);
   }

   /* Enqueue prefix sum kernel */
   size_t ps_global_size = global_size / 2;
   size_t ps_local_size = ps_global_size;
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &ps_global_size,
	   &ps_local_size, 0, NULL, NULL);
   if(err < 0) {
     perror("Couldn't enqueue the kernel");
     exit(1);
   }

   /* Read size of join table */
   err = clEnqueueReadBuffer(queue, carry_buffer, CL_TRUE, 0,
       num_groups * sizeof(cl_uint), carry, 0, NULL, NULL);
   if(err < 0) {
     perror("Couldn't read the buffer");
     exit(1);
   }
   table_joined_size = carry[0];
   printf("Size of joined table: %u\n", table_joined_size);

   /* allocate result table */
   table_joined = malloc(table_joined_size * sizeof(pair));
   if (table_joined == NULL) {
     perror("Couldn't allocate memory");
     exit(1);
   }

   /* allocate result buffer */
   table_joined_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_ALLOC_HOST_PTR, table_joined_size * sizeof(pair), NULL, &err);
   if (err < 0) {
     perror("Couldn't allocate buffer");
     exit(1);
   }
   
   /* Create a kernel */
   kernel = clCreateKernel(p_loop_join, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &table_A_buffer);
   err |= clSetKernelArg(kernel, 1, sizeof(unsigned long long), &table_A_size);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &table_B_buffer);
   err |= clSetKernelArg(kernel, 3, sizeof(unsigned long long), &table_B_size);
   err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &table_joined_partition_sizes_buffer);
   err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &table_joined_partition_offsets_buffer);
   err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &table_joined_buffer);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   /* Enqueue loop join kernel */
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, table_joined_buffer, CL_TRUE, 0, 
         table_joined_size * sizeof(pair), table_joined, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   /* print results */
   printf("Joined table:\n");
   for (i = 0; i != table_joined_size; ++i) {
     printf("(%d %d) ", table_joined[i].first, table_joined[i].second);
   }
   printf("\n");

   /* Deallocate resources */
   clReleaseKernel(kernel);
   clReleaseMemObject(table_joined_partition_sizes_buffer);
   clReleaseMemObject(table_joined_partition_offsets_buffer);
   clReleaseMemObject(table_A_buffer);
   clReleaseMemObject(table_B_buffer);
   clReleaseMemObject(table_joined_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(p_loop_join_count);
   clReleaseProgram(p_loop_join);
   clReleaseContext(context);

   free(table_joined);
   free(table_joined_partition_sizes);
   free(table_joined_partition_offsets);

   return 0;
}
