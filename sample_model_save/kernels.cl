
            #define ADDR(channel, row, col)			channel * (img_size * img_size) + row * img_size + col
            #define ADDR_STRIDE(channel, row, col)	                                                    channel * (output_size * output_size)                                                     + row * output_size + col
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
            
                //printf("This is the Kernel: %d, %d
", filter_id, col);
            
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
                        }
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
        }