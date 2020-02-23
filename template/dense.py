from . import kernel
from . import host


class dense(kernel.kernels):
    def __init__(self, input, output, name, weight_name, bias_name):
        super(self, dense).__init__()
        if not isinstance(input, host.buffer) or not isinstance(output, host.buffer):
            print("Input or Output must be a buffer object!")
            return
        self.name = name
        self.input = input
        self.ouput = output
        self.weight_name = weight_name
        self.bias_name = bias_name

    def write_ip(self):
        s = """
        __kernel void fc_layer(const __global float4* input,     //1D array, should be aligned to 16
                               const __global float4* weight,    //OUT_SIZE X IN_SIZE, should be aligned to 16
                               const __global float* bias,       //OUT_SIZE
                               __global float* output,
                               const int inputSize,             
                               const int outputSize)
        {
            float sum = 0.0;
            const int x = get_global_id(0);
            for (int i = x; i<outputSize; i += get_global_size(0))
            {
                sum = 0.0f;
                for(int j=0; j<inputSize/4 + 1; j++)
                {
                    sum += dot(input[j], weight[i * (inputSize/4) + j]);
                }
                output[i] = sum + bias[i];
            }
        }"""
        return s

    def write_create(self):
        s = "cl_kernel "
        s += host.kernel_template.substitute(kernel_var=self.name, kernel_name='fc_layer')
        return s

    def write_setargs(self):
        s = host.set_arg_template.substitute(kernel_var=self.name, arg_idx=0, DTYPE="cl_mem", var="&" + self.input.name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=1, DTYPE="cl_mem", var="&" + self.weight_name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=2, DTYPE="cl_mem", var="&" + self.bias_name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=3, DTYPE="cl_mem", var="&" + self.output.name) + '\n'
        s += "cl_int param_" + self.name + "_i = %d;\n" % (self.input.size)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=4, DTYPE="cl_int",
                                              var="&" + "param_" + self.name + "_i") + "\n"
        s += "cl_int param_" + self.name + "_o = %d;\n" % (self.output.size)
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=5, DTYPE="cl_int",
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


