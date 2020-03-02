import layer_resolvers
import writer
import lenet
import os

save_root = "./sample_model_save"


model = lenet.Lenet5()
weight_dict, layers_list, ip_list = layer_resolvers.resolve_model(model)

writer.write_ip(os.path.join(save_root, "kernels.cl"), ip_list)
writer.write_weight(os.path.join(save_root, "weights.h"), weight_dict)
writer.write_host(os.path.join(save_root, "host.c"), os.path.join(save_root, "weights.h"), weight_dict, layers_list)