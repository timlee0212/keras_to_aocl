from . import kernel
from . import host


class merge(kernel.kernels):
    def __init__(self, input, output, name, type='max'):
        super(self, merge).__init__()
        if not isinstance(input, host.buffer) or not isinstance(output, host.buffer):
            print("Input or Output must be a buffer object!")
            return
        self.name = name
        self.input = input
        self.ouput = output
        self.type = type

    def write_ip(self):
        #TODO: THIS IS A RESOURCE OPTIMIZED VERSION
        s = '''
        BUFFER_SIZE=256
        __kernel void layer_ops(
            __global float4 *restrict input_tensors,
            __global float4 *restrict output_sum,
            const int input_size_4,                     //The number of float 4
            const char func          //0x01 - add, 0x02 - sub, 0x04 elt-multi, 0x08 avg, 0x0F min & 0x10 max
        )
        {
            float4 shift_buffer[BUFFER_SIZE];
        
        #pragma unroll
            for(int i = 0; i<BUFFER_SIZE;i++)
                shift_buffer[i] = 0;
        
            int load_count = 0;
            //IF VAIRABLE NUM-TENSOR replace NUM_TENSOR with variable division
            while(load_count < input_size_4 + BUFFER_SIZE / NUM_TENSOR)
            {
                //printf("%d\n", load_count);
                //e.g. 4 tensors
                //0 --------------> i
                // 0 | 3 | 2 | 1| 0
                float4 tmp_var = (float4)(0.0, 0.0, 0.0, 0.0);
                for(int tensor=0; tensor< NUM_TENSOR; tensor++)
                {
                    //Only perform computing at the head
                    if(load_count * NUM_TENSOR >= BUFFER_SIZE)
                    {
                        if(func & 0x09)     //Add or Avg
                            tmp_var += shift_buffer[0];
                        else if(func & 0x02)               //Sub
                            tmp_var = shift_buffer[0] - shift_buffer[1];
                        else if(func & 0x04)                //multi
                            tmp_var *= shift_buffer[0];
                        else if(func & 0x0F)
                            tmp_var = fmin(shift_buffer[0], tmp_var);
                        else if(func & 0x10)
                            tmp_var = fmax(shift_buffer[0], tmp_var);
                        #ifdef DEBUG
                        printf("Fetch %.2f\n", shift_buffer[0]);
                        #endif
                    }
                    //Shifting
        #pragma unroll
                    for(int i = 0;i < BUFFER_SIZE - 1; i++)
                    {
                        shift_buffer[i] = shift_buffer[i+1];
                    }
        
                    //Deal with the rest data on Tail
                    if(load_count >= input_size_4 )
                        shift_buffer[BUFFER_SIZE - 1] = 0;
                    else
                        shift_buffer[BUFFER_SIZE - 1] = input_tensors[tensor * input_size_4 ];
                }
                output_sum[load_count - BUFFER_SIZE / NUM_TENSOR] = tmp_var * (REV_NUM_TENSOR * (func & 0x08)); 
                input_tensors++;
                load_count++;
            }
        }
        '''
        return s

    def write_create(self):
        s = "cl_kernel "
        s += host.kernel_template.substitute(kernel_var=self.name,
                                             kernel_name='layer_ops')
        return s

    def write_setargs(self):
        s = host.set_arg_template.substitute(kernel_var=self.name, arg_idx=0, DTYPE="cl_mem",
                                             var="&" + self.input.name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=1, DTYPE="cl_mem",
                                              var="&" + self.weight_name) + '\n'
        s += "cl_uint param_" + self.name + "_1 = %d"%(self.input.size) + ";\n"
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=2, DTYPE="cl_uint",
                                              var="&" + "param_" + self.name + "_1") + '\n'

        s += "char func" + self.name + "=" + self.func + ";\n"
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=3, DTYPE="char",
                                              var="&" + "func_" + self.name) + '\n'

        return s

    def write_release(self):
        s = host.release_buffer.substitute(kernel_var=self.name)
        return s

    def write_enque(self):
        # TODO: Depends on the implementation to decide whether need NDRange or Naive Task
        s =host.enque_kernel_exe.substitute(kernel_var=self.name)
        return s


