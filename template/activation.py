from . import kernel
from . import host

class activation(kernel.kernels):
    def __init__(self, input, output, name, type='relu', params=None):
        super(self, activation).__init__()
        if type not in ['relu', 'elu','sigmoid', 'tanh', 'leakyrelu', 'thresrelu', 'softmax']:
            print(type, " is not supported by current framework.")
            return
        if not isinstance(input, host.buffer) or not isinstance(output, host.buffer):
            print("Input or Output must be a buffer object!")
            return

        self.type = type
        self.params = params
        self.input = input
        self.ouput = output


    def write_ip(self):
        if self.type == 'relu':
            s = """__kernel void relu(__global float *input,
				   __global float *output)
                    {
                        const int x = get_global_id(0);
                        float zero = 0.0;
                    
                        output[x] = fmax(zero, input[x]);
                    }"""
        elif self.type == 'sigmoid':
            s = """__kernel void sigmoid(__global float * input,
					__global float * output)
                {
                    const int x = get_global_id(0);
                
                    //TODO: Optimize the computation
                    output[x] = 1 / ( 1 + exp(-input[x]));
                }"""
        elif self.type == 'elu':
            s = """ __kernel void elu(__global float *input,
				   __global float *output,
				   __constant float *threshold,
				   __constant float *alpha)
                     {
                         const int x = get_global_id(0);
                         if (input[x] > threshold[0])
                             output[x] = input[x];
                         else
                             output[x] = alpha[0] * exp(input[x] - 1);
                     }"""
        elif self.type == 'tanh':
            s = """__kernel void tanh(__global float *input,
                    				   __global float *output)
                    {
                        const int x = get_global_id(0);
                
                        output[x] = 2 / ( 1 + exp(-input[x])) - 1;
                    }"""
        elif self.type == 'leakyrelu':
            s = """ __kernel void leakyrelu(__global float *input, 
                        __global float *output,
						__constant float *threshold,
						__constant float *alpha)
                 {
                     const int x = get_global_id(0);
                
                     if (input[x] > threshold[0])
                         output[x] = input[x];
                     else
                         output[x] = alpha[0] * input[x];
                 }"""
        elif self.type == 'thresrelu':
            s = """ __kernel void thresrelu(__global float *input, 
                         __global float *output,
                         __constant float *threshold)
                     {
                        const int x = get_global_id(0);
                    
                        if (input[x] > threshold[0])
                            output[x] = input[x];
                        else
                            output[x] = 0.0;
                     }"""
        elif self.type == 'softmax':
            s = """__kernel void softmax(__global float * input,
                      __global float * output)
                    {
                        __local float sum, temp[10];
                        const int x = get_local_id(0);
                        temp[x] = exp(input[x]);
                    
                        barrier(CLK_LOCAL_MEM_FENCE);
                        if(get_local_id(0)==0)
                        {
                          for(int i=0; i< get_local_size(0); i++)
                            sum += temp[i];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                        
                        output[x] = temp[x] / sum;	
                    }
                    """
        return s

    def write_create(self):
        s = "cl_kernel "
        s += host.kernel_template.substitute(kernel_var=self.name, kernel_name=self.type)
        return s

    def write_setargs(self):
        s = host.set_arg_template.substitute(kernel_var=self.name, arg_idx=0, DTYPE="cl_mem", var="&"+self.input.name)
        s += "\n" + host.set_arg_template.substitute(kernel_var=self.name, arg_idx=1, DTYPE="cl_mem", var="&"+self.output.name)
        for i in range(len(self.params)):
            s += "\n float param_" + self.name +"_%d = %f\n"%(i, self.params[i])
            s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=i+2, DTYPE="float", var="&"+ "param_" + self.name +"_%d"%(i)) + "\n"

        return s


    def write_release(self):
        s = host.release_buffer.substitute(kernel_var = self.name)
        return s

    def write_enque(self):
        # TODO: Depends on the implementation to decide whether need NDRange or Naive Task
        s = "cl_uint gl_size_" + self.name + " = 128\n"
        s += host.enque_ndrange.substitute(kernel_var=self.name, gl_size="gl_size_"+ self.name, local_size= "NULL")
        return s