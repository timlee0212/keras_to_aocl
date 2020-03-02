import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import template

def resolve_model(model):
    if not isinstance(model, keras.models.Sequential):
        print("Only Sequential is supported currently!")
        return

    layer_id = 0
    layers_list = []
    ip_list = []
    weight_dict = {}
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            weight_dict["fc_%d_weight"%layer_id] = layer.get_weights()[0];
            weight_dict["fc_%d_bias"%layer_id] = layer.get_weights()[1];
            if len(layers_list) == 0:   #The first Layer
                input_buf = template.host.buffer(layer.get_input_shape_at(0)[1], "fc_%d_in"%layer_id)
            else:
                input_buf = layers_list[-1].output
            output_buf = template.host.buffer(layer.get_output_shape_at(0)[1], "fc_%d_out"%layer_id)

            layers_list.append(template.dense.dense(input_buf, output_buf,
                                                    "fc_%d"%layer_id, "fc_%d_weight"%layer_id ,"fc_%d_bias"%layer_id))

            ip = layers_list[-1].write_ip()
            if ip not in ip_list:
                ip_list.append(ip)
        elif isinstance(layer, layers.Activation):
            input_buf = layers_list[-1].output
            output_buf = template.host.buffer(layer.get_output_shape_at(0)[1], "act_%d_out" % layer_id)
            params = None
            if isinstance(layer, layers.ReLU):
                types = 'relu'
            elif isinstance(layer)

            layers_list.append(template.activation.activation(input_buf, output_buf, "act_%d_out" % layer_id, types, params))

        layer_id += 1