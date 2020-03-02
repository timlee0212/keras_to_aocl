import os
import template

def write_ip(ip_file_name, ip_list):
    f = open(ip_file_name, "w")
    for ip in ip_list:
        f.write(ip)
    f.close()

#Save Weight into C_header file
def write_weight(weight_file_name, weight_dict):
    f = open(weight_file_name, 'w')
    for key, value in weight_dict.items():
        f.write(key+"={")
        w = value.flatten()
        for val in w:
            f.write(str(val)+",")
        f.write("};\n")
    f.close()

def write_host(host_c_name, weight_file_name, weight_dict, layers_list):
    f = open(host_c_name, 'w')
    f.write(template.host.file_header.substitute(weight_file = weight_file_name))
    #Initialize Weights
    weights = []
    for key, value in weight_dict.items():
        weight_buf = template.host.buffer(value.shape, key, mode='r')
        weights.append(weight_buf)
        f.write(weight_buf.write_create())
        f.write(weight_buf.write_bufwrite(key))

    f.write(layers_list[0].input.write_create())
    #Initilize Buffers
    for layer in layers_list:
        f.write(layer.output.write_create())

    #Initilize Kernels
    for layer in layers_list:
        f.write(layer.write_create())

    #Execute Kernels
    for layer in layers_list:
        f.write(layer.write_setargs())
        f.write(layer.write_enque())


    #Ending
    f.write(layers_list[0].input.write_release())
    for layer in layers_list:
        f.write(layer.output.write_release())
    for weight in weights:
        f.write(weight.write_release())
    for layer in layers_list:
        f.write(layer.write_release())
    f.write(template.host.end_template)
