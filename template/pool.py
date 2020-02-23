from . import kernel
from . import host


class pool(kernel.kernels):
    def __init__(self, input, output, name, params, type='max'):
        super(self, pool).__init__()
        if not isinstance(input, host.buffer) or not isinstance(output, host.buffer):
            print("Input or Output must be a buffer object!")
            return
        self.name = name
        self.input = input
        self.ouput = output
        self.type = type
        self.params = params
    def write_ip(self):
        if self.type=='max':
            s = """__kernel void maxPool2D(const __global float *input,
						__global float * output,
						__constant int *params)
                {
                    //get the position of the current work item
                    const int x = get_global_id(0);
                    const int y = get_global_id(1);
                
                    //get the true position in the image
                    const int xidx = params[4] * x;
                    const int yidx = params[5] * y;
                
                    float maxval = -3.5e20;
                
                    for(int r = 0; r < params[1]; r++)
                    {
                        for(int c = 0; c < params[0]; c++)
                        {
                            const int idxin = ((yidx + r) * params[2]) + xidx + c;
                            maxval = fmax(maxval, input[idxin]);
                        }
                    }"""
        elif self.type == 'avg':
            s ="""
                __kernel void avgPool2D(const __global float *input,
						__global float *output,
						__constant int *params)
                        {
                            __private sum = 0.0;
                        
                            //get the position of the current work item
                            const int x = get_global_id(0);
                            const int y = get_global_id(1);
                        
                            //get the true position in the image
                            const int xidx = params[4] * x;
                            const int yidx = params[5] * y;
                        
                            for (int r = 0; r < params[1]; r++)
                            {
                                for (int c = 0; c < params[0]; c++)
                                {
                                    const int idxin = ((yidx + r) * params[2]) + xidx + c;
                                    sum += input[idxin];
                                }
                            }
                        
                            output[(y * (params[2] / params[4])) + x] = sum;
                        }"""
        return s

    def write_create(self):
        s = "cl_kernel "
        s += host.kernel_template.substitute(kernel_var=self.name, kernel_name='maxPool2D' if self.type=='max' else 'avgPool2D')
        return s

    def write_setargs(self):
        s = host.set_arg_template.substitute(kernel_var=self.name, arg_idx=0, DTYPE="cl_mem", var="&" + self.input.name) + '\n'
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=1, DTYPE="cl_mem", var="&" + self.weight_name) + '\n'
        s += "cl_uint param_" + self.name + "[6] = {%d, %d, %d, %d, %d, %d}\n" % (self.params[0], self.params[1],
                                                                                  self.params[2], self.params[3],
                                                                                  self.params[4], self.params[5]) +";\n"
        s += host.set_arg_template.substitute(kernel_var=self.name, arg_idx=2, DTYPE="cl_uint*",
                                              var="&" + "param_" + self.name) + '\n'

        return s

    def write_release(self):
        s = host.release_buffer.substitute(kernel_var=self.name)
        return s

    def write_enque(self):
        # TODO: Depends on the implementation to decide whether need NDRange or Naive Task
        s = "cl_uint gl_size_" + self.name + "[2] = {128， 128}\n"
        s += host.enque_ndrange.substitute(kernel_var=self.name, gl_size="gl_size_" + self.name, local_size="NULL")
        return s


