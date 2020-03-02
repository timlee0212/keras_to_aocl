
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include "./sample_model_save\weights.h"

#include <CL/cl.h>

#include "ocl_util.h"
#include "timer.h"
#include "weight.h"

using namespace std;
using namespace ocl_util;

const char *vendor_name = "Intel";
#define DEVICE_TYPE CL_DEVICE_TYPE_ACCELERATOR

int main(int argc, char *argv[])
{
	cl_int status;

	cl_uint num_devices = 0;
	cl_platform_id platform_id = NULL;
	cl_context context = NULL;
	cl_program program = NULL;

	scoped_array<cl_device_id> device;

	Timer t;  // Timer used for performance measurement
	float time;

	if (argc != 2)
	{
		printf("Error: wrong commad format, usage:
");
		printf("%s <binaryfile>
", argv[0]);
		return EXIT_FAILURE;
	}

	printf("***************************************************
");
	printf("Manual Test of LeNet-5 
");
	printf("***************************************************
");

	// Connect to the desired platform
	platform_id = findPlatform(vendor_name);
	if(platform_id == NULL) 
	{
		printf("ERROR: Unable to find the desired OpenCL platform.
");
		return false;
	}

	// Query the available OpenCL device
	device.reset(getDevices(platform_id, DEVICE_TYPE, &num_devices));
	printf("
Platform: %s
", getPlatformName(platform_id).c_str());
	printf("Using %d device(s)
", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) 
	{
		printf("  Device %d: %s
", i, getDeviceName(device[i]).c_str());
		displayDeviceInfo(device[i]);
	}


	// Create the context.
	context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
	checkError(status, "Failed to create context");

	// Create Program Objects
	char *kernel_file_name=argv[1];

	// Create the program for all device. All devices execute the same kernel.
	program = createProgramFromFile(context, (const char *) kernel_file_name, device, num_devices);
	
	queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
cl_mem conv_0_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, 54* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, conv_0_weight, CL_TRUE, 0, 54 * float,			                            conv_0_weight, 0, NULL, NULL);
cl_mem conv_0_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, 6* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, conv_0_bias, CL_TRUE, 0, 6 * float,			                            conv_0_bias, 0, NULL, NULL);
cl_mem conv_2_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, 864* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, conv_2_weight, CL_TRUE, 0, 864 * float,			                            conv_2_weight, 0, NULL, NULL);
cl_mem conv_2_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, 16* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, conv_2_bias, CL_TRUE, 0, 16 * float,			                            conv_2_bias, 0, NULL, NULL);
cl_mem fc_5_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, 69120* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, fc_5_weight, CL_TRUE, 0, 69120 * float,			                            fc_5_weight, 0, NULL, NULL);
cl_mem fc_5_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, 120* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, fc_5_bias, CL_TRUE, 0, 120 * float,			                            fc_5_bias, 0, NULL, NULL);
cl_mem fc_6_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, 10080* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, fc_6_weight, CL_TRUE, 0, 10080 * float,			                            fc_6_weight, 0, NULL, NULL);
cl_mem fc_6_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, 84* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, fc_6_bias, CL_TRUE, 0, 84 * float,			                            fc_6_bias, 0, NULL, NULL);
cl_mem fc_7_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, 840* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, fc_7_weight, CL_TRUE, 0, 840 * float,			                            fc_7_weight, 0, NULL, NULL);
cl_mem fc_7_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, 10* sizeof(float), NULL, &status);
status = clEnqueueWriteBuffer(queue, fc_7_bias, CL_TRUE, 0, 10 * float,			                            fc_7_bias, 0, NULL, NULL);
cl_mem conv_0_in = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024* sizeof(float), NULL, &status);cl_mem conv_0_out = clCreateBuffer(context, CL_MEM_READ_WRITE, 5400* sizeof(float), NULL, &status);
cl_mem pool_1_out = clCreateBuffer(context, CL_MEM_READ_WRITE, 1350* sizeof(float), NULL, &status);
cl_mem conv_2_out = clCreateBuffer(context, CL_MEM_READ_WRITE, 2704* sizeof(float), NULL, &status);
cl_mem pool_3_out = clCreateBuffer(context, CL_MEM_READ_WRITE, 576* sizeof(float), NULL, &status);
cl_mem fc_5_out = clCreateBuffer(context, CL_MEM_READ_WRITE, 120* sizeof(float), NULL, &status);
cl_mem fc_6_out = clCreateBuffer(context, CL_MEM_READ_WRITE, 84* sizeof(float), NULL, &status);
cl_mem fc_7_out = clCreateBuffer(context, CL_MEM_READ_WRITE, 10* sizeof(float), NULL, &status);
cl_kernel conv_0 = clCreateKernel(program, conv2D, &status);
cl_kernel pool_1 = clCreateKernel(program, avgPool2D, &status);
cl_kernel conv_2 = clCreateKernel(program, conv2D, &status);
cl_kernel pool_3 = clCreateKernel(program, avgPool2D, &status);
cl_kernel fc_5 = clCreateKernel(program, fc_layer, &status);
cl_kernel fc_6 = clCreateKernel(program, fc_layer, &status);
cl_kernel fc_7 = clCreateKernel(program, fc_layer, &status);
clSetKernelArg(conv_0, 0, sizeof(cl_mem), &conv_0_in);
clSetKernelArg(conv_0, 1, sizeof(cl_mem), &conv_0_weight);
clSetKernelArg(conv_0, 2, sizeof(cl_mem), &conv_0_bias);
clSetKernelArg(conv_0, 3, sizeof(cl_mem), &conv_0_out);
cl_short param_conv_0_1 = 1;
clSetKernelArg(conv_0, 4, sizeof(cl_short), &param_conv_0_1);
cl_short param_conv_0_2 = 1;
clSetKernelArg(conv_0, 5, sizeof(cl_short), &param_conv_0_2);
cl_short param_conv_0_3 = 6;
clSetKernelArg(conv_0, 6, sizeof(cl_short), &param_conv_0_3);
cl_short param_conv_0_4 = 3;
clSetKernelArg(conv_0, 7, sizeof(cl_short), &param_conv_0_4);
cl_int param_conv_0_5 = 32;
clSetKernelArg(conv_0, 8, sizeof(cl_int), &param_conv_0_5);
cl_short param_conv_0_6 = 1;
clSetKernelArg(conv_0, 9, sizeof(cl_short), &param_conv_0_i);
cl_int param_conv_0_7 = 30;
clSetKernelArg(conv_0, 10, sizeof(cl_int), &param_conv_0_o);

cl_uint gl_size_conv_0 = {256, 256}
clEnqueueNDRangeKernel(queue, conv_0, 3, NULL, gl_size_conv_0, NULL, 0, NULL, NULL);
clSetKernelArg(pool_1, 0, sizeof(cl_mem), &conv_0_out);
clSetKernelArg(pool_1, 1, sizeof(cl_mem), &pool_1_out);
cl_uint param_pool_1[6] = {2, 2, 15, 15, 2, 2}
;
clSetKernelArg(pool_1, 2, sizeof(cl_uint*), &param_pool_1);

cl_uint gl_size_pool_1[2] = {128£¬ 128}
clEnqueueNDRangeKernel(queue, pool_1, 3, NULL, gl_size_pool_1, NULL, 0, NULL, NULL);
clSetKernelArg(conv_2, 0, sizeof(cl_mem), &pool_1_out);
clSetKernelArg(conv_2, 1, sizeof(cl_mem), &conv_2_weight);
clSetKernelArg(conv_2, 2, sizeof(cl_mem), &conv_2_bias);
clSetKernelArg(conv_2, 3, sizeof(cl_mem), &conv_2_out);
cl_short param_conv_2_1 = 1;
clSetKernelArg(conv_2, 4, sizeof(cl_short), &param_conv_2_1);
cl_short param_conv_2_2 = 1;
clSetKernelArg(conv_2, 5, sizeof(cl_short), &param_conv_2_2);
cl_short param_conv_2_3 = 16;
clSetKernelArg(conv_2, 6, sizeof(cl_short), &param_conv_2_3);
cl_short param_conv_2_4 = 3;
clSetKernelArg(conv_2, 7, sizeof(cl_short), &param_conv_2_4);
cl_int param_conv_2_5 = 15;
clSetKernelArg(conv_2, 8, sizeof(cl_int), &param_conv_2_5);
cl_short param_conv_2_6 = 6;
clSetKernelArg(conv_2, 9, sizeof(cl_short), &param_conv_2_i);
cl_int param_conv_2_7 = 13;
clSetKernelArg(conv_2, 10, sizeof(cl_int), &param_conv_2_o);

cl_uint gl_size_conv_2 = {256, 256}
clEnqueueNDRangeKernel(queue, conv_2, 3, NULL, gl_size_conv_2, NULL, 0, NULL, NULL);
clSetKernelArg(pool_3, 0, sizeof(cl_mem), &conv_2_out);
clSetKernelArg(pool_3, 1, sizeof(cl_mem), &pool_3_out);
cl_uint param_pool_3[6] = {2, 2, 6, 6, 2, 2}
;
clSetKernelArg(pool_3, 2, sizeof(cl_uint*), &param_pool_3);

cl_uint gl_size_pool_3[2] = {128£¬ 128}
clEnqueueNDRangeKernel(queue, pool_3, 3, NULL, gl_size_pool_3, NULL, 0, NULL, NULL);
clSetKernelArg(fc_5, 0, sizeof(cl_mem), &pool_3_out);
clSetKernelArg(fc_5, 1, sizeof(cl_mem), &fc_5_weight);
clSetKernelArg(fc_5, 2, sizeof(cl_mem), &fc_5_bias);
clSetKernelArg(fc_5, 3, sizeof(cl_mem), &fc_5_out);
cl_int param_fc_5_i = 576;
clSetKernelArg(fc_5, 4, sizeof(cl_int), &param_fc_5_i);
cl_int param_fc_5_o = 120;
clSetKernelArg(fc_5, 5, sizeof(cl_int), &param_fc_5_o);

cl_uint gl_size_fc_5 = 128
clEnqueueNDRangeKernel(queue, fc_5, 3, NULL, gl_size_fc_5, NULL, 0, NULL, NULL);
clSetKernelArg(fc_6, 0, sizeof(cl_mem), &fc_5_out);
clSetKernelArg(fc_6, 1, sizeof(cl_mem), &fc_6_weight);
clSetKernelArg(fc_6, 2, sizeof(cl_mem), &fc_6_bias);
clSetKernelArg(fc_6, 3, sizeof(cl_mem), &fc_6_out);
cl_int param_fc_6_i = 120;
clSetKernelArg(fc_6, 4, sizeof(cl_int), &param_fc_6_i);
cl_int param_fc_6_o = 84;
clSetKernelArg(fc_6, 5, sizeof(cl_int), &param_fc_6_o);

cl_uint gl_size_fc_6 = 128
clEnqueueNDRangeKernel(queue, fc_6, 3, NULL, gl_size_fc_6, NULL, 0, NULL, NULL);
clSetKernelArg(fc_7, 0, sizeof(cl_mem), &fc_6_out);
clSetKernelArg(fc_7, 1, sizeof(cl_mem), &fc_7_weight);
clSetKernelArg(fc_7, 2, sizeof(cl_mem), &fc_7_bias);
clSetKernelArg(fc_7, 3, sizeof(cl_mem), &fc_7_out);
cl_int param_fc_7_i = 84;
clSetKernelArg(fc_7, 4, sizeof(cl_int), &param_fc_7_i);
cl_int param_fc_7_o = 10;
clSetKernelArg(fc_7, 5, sizeof(cl_int), &param_fc_7_o);

cl_uint gl_size_fc_7 = 128
clEnqueueNDRangeKernel(queue, fc_7, 3, NULL, gl_size_fc_7, NULL, 0, NULL, NULL);
clReleaseMemObject(conv_0_in);
clReleaseMemObject(conv_0_out);
clReleaseMemObject(pool_1_out);
clReleaseMemObject(conv_2_out);
clReleaseMemObject(pool_3_out);
clReleaseMemObject(fc_5_out);
clReleaseMemObject(fc_6_out);
clReleaseMemObject(fc_7_out);
clReleaseMemObject(conv_0_weight);
clReleaseMemObject(conv_0_bias);
clReleaseMemObject(conv_2_weight);
clReleaseMemObject(conv_2_bias);
clReleaseMemObject(fc_5_weight);
clReleaseMemObject(fc_5_bias);
clReleaseMemObject(fc_6_weight);
clReleaseMemObject(fc_6_bias);
clReleaseMemObject(fc_7_weight);
clReleaseMemObject(fc_7_bias);
clReleaseKernel(conv_0);
clReleaseKernel(pool_1);
clReleaseKernel(conv_2);
clReleaseKernel(pool_3);
clReleaseKernel(fc_5);
clReleaseKernel(fc_6);
clReleaseKernel(fc_7);

    clReleaseCommandQueue(queue);
	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}
	
    return 0;
}
