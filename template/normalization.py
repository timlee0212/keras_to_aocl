from . import kernel
from . import host


class batch_norm(kernel.kernels):
    def __init__(self, input, output, name, par_a_name, par_b_name):
        super(self, dense).__init__()
        if not isinstance(input, host.buffer) or not isinstance(output, host.buffer):
            print("Input or Output must be a buffer object!")
            return
        self.name = name
        self.input = input
        self.ouput = output
        self.par_a_name = par_a_name
        self.par_b_name = par_b_name

    def write_ip(self):
        s = """
            #define BUFFER_SIZE     10
            //Optimized Version of the Batch Normalization
            //Single Work-Item Kernel
            //Using Shift Register
            __kernel void batch_norm(
                __global float4* restrict input,
                __global float4* restrict output,
                __global float4* restrict par_a,     //gamma / sqrt(sigma^2 + epsilon)
                __global float4* restrict par_b,    //mu / sqrt(sigma^2 + epsilon) + beta
                const int input_size)
            {
            
                float4 input_buffer[BUFFER_SIZE], 
                        par_a_buf[BUFFER_SIZE], par_b_buf[BUFFER_SIZE];
            
            #pragma unroll
                for(int i=0; i<BUFFER_SIZE; i++)
                {       
                    input_buffer[i] = 0;
                    par_a_buf[i] = 0;
                    par_b_buf[i] = 0;
                }
            
                int load_count = 0;
                while(load_count < input_size + BUFFER_SIZE )
                {
                    if(load_count >= BUFFER_SIZE)
                        output[load_count - BUFFER_SIZE] = mad(par_a_buf[0], input_buffer[0], par_b_buf[0]);
            
                    #pragma unroll
                    for(int i=0; i<BUFFER_SIZE-1; i++)
                    {
                        input_buffer[i] = input_buffer[i+1];
                        par_a_buf[i] = par_a_buf[i+1];
                        par_b_buf[i] = par_b_buf[i+1];
                    }
            
                    if(load_count < input_size)
                    {
                        input_buffer[BUFFER_SIZE - 1] = input[load_count];
                        par_a_buf[BUFFER_SIZE - 1] = par_a[load_count];
                        par_b_buf[BUFFER_SIZE - 1] = par_b[load_count];
                    }
                    else
                    {
                        input_buffer[BUFFER_SIZE - 1] = 0;
                        par_a_buf[BUFFER_SIZE - 1] = 0;
                        par_b_buf[BUFFER_SIZE - 1] = 0;
                    }
                    
                    load_count++;
                }
            }
            """
        return s

    def write_create(self):
        s = "cl_kernel "
        s += host.kernel_template.substitute(kernel_var=self.name, kernel_name='fc_layer')
        return s

    def write_setargs(self):
        s = host.set_arg_template.substitute(kernel_var=self.name, arg_idx=0, DTYPE="cl_mem", var="&" + self.input.name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=1, DTYPE="cl_mem", var="&" + self.output.name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=2, DTYPE="cl_mem", var="&" + self.par_a_name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=3, DTYPE="cl_mem", var="&" + self.par_b_name) + '\n'
        s += "cl_int param_" + self.name + "_i = %d;\n" % (self.input.size)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=4, DTYPE="cl_int",
                                              var="&" + "param_" + self.name + "_i") + "\n"
        return s

    def write_release(self):
        s = host.release_buffer.substitute(kernel_var=self.name)
        return s

    def write_enque(self):
        # TODO: Depends on the implementation to decide whether need NDRange or Naive Task
        s = host.enque_kernel_exe.substitute(kernel_var=self.name)
        return s


