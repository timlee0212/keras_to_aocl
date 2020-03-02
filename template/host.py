########
# This File contains template for Hardware IP and software programs
########

from string import Template
import numpy as np

##================ Software Template ==============================

file_header = \
Template(
"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include "${weight_file}"

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
		printf("Error: wrong commad format, usage:\n");
		printf("%s <binaryfile>\n", argv[0]);
		return EXIT_FAILURE;
	}

	printf("***************************************************\n");
	printf("Manual Test of LeNet-5 \n");
	printf("***************************************************\n");

	// Connect to the desired platform
	platform_id = findPlatform(vendor_name);
	if(platform_id == NULL) 
	{
		printf("ERROR: Unable to find the desired OpenCL platform.\n");
		return false;
	}

	// Query the available OpenCL device
	device.reset(getDevices(platform_id, DEVICE_TYPE, &num_devices));
	printf("\nPlatform: %s\n", getPlatformName(platform_id).c_str());
	printf("Using %d device(s)\n", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) 
	{
		printf("  Device %d: %s\n", i, getDeviceName(device[i]).c_str());
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
""")

####### Create Kernels
kernel_template = Template("${kernel_var} = clCreateKernel(program, ${kernel_name}, &status);")

###### Buffers
buffer_reado_template = Template("${buf_name} = clCreateBuffer(context, CL_MEM_READ_ONLY, ${buf_size}* sizeof(${DTYPE}), NULL, &status);")
buffer_writeo_template = Template("${buf_name} = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ${buf_size}* sizeof(${DTYPE}), NULL, &status);")
buffer_rw_template = Template("${buf_name} = clCreateBuffer(context, CL_MEM_READ_WRITE, ${buf_size}* sizeof(${DTYPE}), NULL, &status);")

###### Enqueue
enque_buffer_write = Template(
			"status = clEnqueueWriteBuffer(queue, ${buf_name}, CL_TRUE, 0, ${buffer_size} * ${DTYPE},\
			                            ${host_name}, 0, NULL, NULL);")

### Args
set_arg_template = Template("clSetKernelArg(${kernel_var}, ${arg_idx}, sizeof(${DTYPE}), ${var});")
enque_kernel_exe= Template("clEnqueueTask(queue, ${kernel_var}, 0, NULL, NULL);")

enque_ndrange = Template("clEnqueueNDRangeKernel(queue, ${kernel_var}, 3, NULL, ${gl_size}, ${local_size}, 0, NULL, NULL);")

release_kernel = Template("clReleaseKernel(${kernel_var});")
release_buffer = Template("clReleaseMemObject(${buffer_name});")

end_template = """
    clReleaseCommandQueue(queue);
	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}
	
    return 0;
}
"""

##================== End Software Template ========================

class buffer():
	def __init__(self, size, name, mode='rw', dtype="float"):
		if dtype not in ['double', "float", "int", "uint", "short", "ushort", "char"]:
			print("No a valid dtype!")
		self.dtype = dtype
		self.name = name
		self.size = np.array(size)
		self.mode = mode

		if len(self.size) > 1:
			self.size = np.prod(self.size)
		self.size = self.size[0]

	def write_create(self):
		s = "cl_mem "
		if self.mode=='r':
			s += buffer_reado_template.substitute(buf_name = self.name, bufsize=self.size, DTYPE=self.dtype)
		elif self.mode=='w':
			s += buffer_writeo_template.substitute(buf_name=self.name, bufsize=self.size, DTYPE=self.dtype)
		else:
			s += buffer_rw_template.substitute(buf_name = self.name, bufsize=self.size, DTYPE=self.dtype)

		return s

	def write_bufwrite(self, host_ptr):
		return enque_buffer_write.substitute(buf_name=self.name, buffer_size=self.size, DTYPE=self.dtype,
											 host_name=host_ptr)

	def write_release(self):
		return release_buffer.substitute(buffer_name=self.name)


