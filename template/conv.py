from . import kernel
from . import host


class conv1D(kernel.kernels):
    def __init__(self, input, output, name, weight_name, bias_name,
                 stride, padding, num_filter, filter_size):
        if not isinstance(input, host.buffer) or not isinstance(output, host.buffer):
            print("Input or Output must be a buffer object!")
            return
        self.name = name
        self.input = input
        self.output = output
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.stride = stride
        self.padding = padding,
        self.num_filter = num_filter
        self.filter_size = filter_size
    def write_ip(self):
        s = """
            #define ADDR(channel, row, col)			channel * (img_size * img_size) + row * img_size + col
            #define ADDR_STRIDE(channel, row, col)	\
                                                    channel * (output_size * output_size)\
                                                     + row * output_size + col
            
            __kernel void conv1D(
                __global const float *restrict input_seq,
                __global const float *restrict filter,
                __global const float *restrict bias,
                __global float* restrict output_result,
                const short stride, const short padding, /*0 - No Padding, 1 - Zero Padding*/ 
                const short num_filter, const short filter_size,
                const int seq_size, const int output_size
            )
            {
                //Input Shape is:	Channels x Lengths
                //Filter Shape is:	num_filters x Lengths
                //Output Shape is:	Num Filters x length/stride
                const int filter_id = get_global_id(0);
            
                float curr_bias = bias[filter_id];
                __local float filterCache[256];
            
            #pragma unroll
                for(int filter_pos = 0; filter_pos < filter_size; filter_pos++)
                {
                    filterCache[filter_pos] = filter[filter_id * filter_size + filter_pos];
                }
            
            #pragma ivdep
                for(int pos =0; pos<seq_size; pos++)
                {
                    float tmp_sum = curr_bias;
                    #pragma unroll
                    for(int flt_pos = 0; flt_pos < filter_size; flt_pos++)
                    {
                        int addr = pos * stride + flt_pos - padding * (filter_size >> 1 );
                        
                        if(addr <= 0 || addr >= (seq_size + (filter_size>>1)))
                            continue;
                        
                        tmp_sum += input_seq[addr] * filterCache[flt_pos];
                    }
            
                    output_result[pos] = tmp_sum;
                }
            
            }
        """
        return s

    def write_create(self):
        s = "cl_kernel "
        s += host.kernel_template.substitute(kernel_var=self.name, kernel_name='conv1D')
        return s

    def write_setargs(self):
        s = host.set_arg_template.substitute(kernel_var=self.name, arg_idx=0, DTYPE="cl_mem", var="&" + self.input.name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=1, DTYPE="cl_mem", var="&" + self.weight_name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=2, DTYPE="cl_mem", var="&" + self.bias_name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=3, DTYPE="cl_mem", var="&" + self.output.name) + '\n'
        s += "cl_short param_" + self.name + "_1 = %d;\n" % (self.stride)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=4, DTYPE="cl_short",
                                              var="&" + "param_" + self.name + "_1") + "\n"
        s += "cl_short param_" + self.name + "_2 = %d;\n" % (self.padding)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=5, DTYPE="cl_short",
                                              var="&" + "param_" + self.name + "_2") + "\n"
        s += "cl_short param_" + self.name + "_3 = %d;\n" % (self.num_filter)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=6, DTYPE="cl_short",
                                              var="&" + "param_" + self.name + "_3") + "\n"
        s += "cl_short param_" + self.name + "_4 = %d;\n" % (self.filter_size)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=7, DTYPE="cl_short",
                                              var="&" + "param_" + self.name + "_4") + "\n"
        s += "cl_int param_" + self.name + "_5 = %d;\n" % (self.input.size)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=8, DTYPE="cl_int",
                                              var="&" + "param_" + self.name + "_5") + "\n"
        s += "cl_int param_" + self.name + "_6 = %d;\n" % (self.output.size)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=9, DTYPE="cl_int",
                                              var="&" + "param_" + self.name + "_o") + "\n"

        return s

    def write_release(self):
        s = host.release_buffer.substitute(kernel_var=self.name)
        return s

    def write_enque(self):
        # TODO: Depends on the implementation to decide whether need NDRange or Naive Task
        s = "cl_uint gl_size_" + self.name + " = 128\n"
        s += host.enque_ndrange.substitute(kernel_var=self.name, gl_size="gl_size_" + self.name, local_size="NULL")
        return s


class conv2D(kernel.kernels):
    def __init__(self, input, output, name, weight_name, bias_name,
                 stride, padding, num_filter, filter_size, img_size, input_channels, output_size):
        if not isinstance(input, host.buffer) or not isinstance(output, host.buffer):
            print("Input or Output must be a buffer object!")
            return
        self.name = name
        self.input = input
        self.output = output
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.stride = stride
        self.padding = padding,
        self.num_filter = num_filter
        self.filter_size = filter_size
        self.img_size = img_size
        self.input_channels = input_channels
        self.output_size = output_size

    def write_ip(self):
        s = """
            #define ADDR(channel, row, col)			channel * (img_size * img_size) + row * img_size + col
            #define ADDR_STRIDE(channel, row, col)	\
                                                    channel * (output_size * output_size)\
                                                     + row * output_size + col
            __kernel void conv2D(
                __global const float *restrict input_img,
                __global const float *restrict filter,
                __global const float *restrict bias,
                __global float* restrict output_result,
                const short stride, const short padding, /*0 - No Padding, 1 - Zero Padding*/ 
                const short num_filter, const short filter_size,
                const short img_size, const short input_channels,
                const short output_size)
            {
                //Input Shape is:	Channels x width x height
                //Filter Shape is:	Num_Filters x width x height x Channels
                //Output Shape is:	Num Filters x width/stride x height/stride
                const int filter_id = get_global_id(0);
                const int col = get_global_id(1);
                __local float filterCache[256]
            
                __local float curr_bias;
            
                int filterSize2 = filter_size * filter_size;
            
                //printf("This is the Kernel: %d, %d\n", filter_id, col);
            
                if(col == 0)    //Initializer Work-Item
                { 
                    curr_bias = bias[filter_id];
                    for(int chan = 0; chan < input_channels; chan++)
                    {
                        for(int filter_pos = 0; filter_pos < filterSize2; filter_pos++)
                        {
                            filterCache[chan* filterSize2 + filter_pos] = filter[filter_id * (filterSize2 * input_channels) + 
                                                                                   chan * filterSize2 + filter_pos];
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int col = 0; col < img_size / stride; col++)
                {
                    for (int row = 0; row < output_size; row++)
                    {
                        float tmp_sum = curr_bias;
                        for (int chan = 0; chan < input_channels; chan++)
                        {
                            for (int flt_r = 0; flt_r < filter_size; flt_r++)
                            {
                                for (int flt_c = 0; flt_c < filter_size; flt_c++)
                                {
                                    int row_addr = row * stride + flt_r - padding * (filter_size >> 1);
                                    int col_addr = col * stride + flt_c - padding * (filter_size >> 1);
                                    
                                    if (row_addr < 0 || col_addr < 0 || row_addr >= img_size || col_addr >= img_size)	//Padding Area
                                        continue;
            
                                    tmp_sum += input_img[ADDR(chan, row_addr, col_addr)] * filterCache[chan * filterSize2 + flt_r * filter_size + flt_c];
                                }
                            }
                        }
            
                        //TODO: ACTIVATION FUNCTIONS
                         output_result[ADDR_STRIDE(filter_id, row, col)] = tmp_sum;
            
                    }
                }
            }
        """
        return s

    def write_create(self):
        s = "cl_kernel "
        s += host.kernel_template.substitute(kernel_var=self.name, kernel_name='conv2D')
        return s

    def write_setargs(self):
        s = host.set_arg_template.substitute(kernel_var=self.name, arg_idx=0, DTYPE="cl_mem",
                                             var="&" + self.input.name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=1, DTYPE="cl_mem",
                                              var="&" + self.weight_name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=2, DTYPE="cl_mem",
                                              var="&" + self.bias_name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=3, DTYPE="cl_mem",
                                              var="&" + self.output.name) + '\n'
        s += "cl_short param_" + self.name + "_1 = %d;\n" % (self.stride[0])
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=4, DTYPE="cl_short",
                                              var="&" + "param_" + self.name + "_1") + "\n"
        s += "cl_short param_" + self.name + "_2 = %d;\n" % (self.padding)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=5, DTYPE="cl_short",
                                              var="&" + "param_" + self.name + "_2") + "\n"
        s += "cl_short param_" + self.name + "_3 = %d;\n" % (self.num_filter)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=6, DTYPE="cl_short",
                                              var="&" + "param_" + self.name + "_3") + "\n"
        s += "cl_short param_" + self.name + "_4 = %d;\n" % (self.filter_size)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=7, DTYPE="cl_short",
                                              var="&" + "param_" + self.name + "_4") + "\n"
        s += "cl_int param_" + self.name + "_5 = %d;\n" % (self.img_size)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=8, DTYPE="cl_int",
                                              var="&" + "param_" + self.name + "_5") + "\n"
        s += "cl_short param_" + self.name + "_6 = %d;\n" % (self.input_channels)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=9, DTYPE="cl_short",
                                              var="&" + "param_" + self.name + "_i") + "\n"
        s += "cl_int param_" + self.name + "_7 = %d;\n" % (self.output_size)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=10, DTYPE="cl_int",
                                              var="&" + "param_" + self.name + "_o") + "\n"

        return s

    def write_release(self):
        s = host.release_kernel.substitute(kernel_var=self.name)
        return s

    def write_enque(self):
        # TODO: Depends on the implementation to decide whether need NDRange or Naive Task
        s = "cl_uint gl_size_" + self.name + " = {256, 256};\n"
        s += host.enque_ndrange.substitute(kernel_var=self.name, gl_size="gl_size_" + self.name, local_size="NULL")
        return s

