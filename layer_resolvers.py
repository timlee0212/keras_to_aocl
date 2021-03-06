import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import template.host
import template.dense
import template.normalization
import template.conv
import template.activation
import template.pool

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
            weight_dict, layers_list = _add_dense(layer, weight_dict, layers_list, layer_id)
            # if layer.activation is not None:
            #     layer_id += 1
            #     layers_list = _add_act(layer.activation, layers_list, layer_id)

        elif isinstance(layer, layers.Activation):
            layers_list = _add_act(layer, layers_list, layer_id)

        elif isinstance(layer, layers.MaxPool2D) or isinstance(layer, layers.AvgPool2D):
            layers_list = _add_pool(layer, layers_list,layer_id)

        elif isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Conv1D):
            weight_dict, layers_list = _add_conv(layer, weight_dict, layers_list, layer_id)

        elif isinstance(layer, layers.BatchNormalization):
            weight_dict, layers_list = _add_bn(layer, weight_dict, layers_list, layer_id)

        else:
            print("%s Layer are not supported yet." % layer.name)


        ip = layers_list[-1].write_ip()
        if ip not in ip_list:
            ip_list.append(ip)

        layer_id += 1

    return weight_dict, layers_list, ip_list

def _add_dense(layer, weight_dict, layers_list, layer_id):
    weight_dict["fc_%d_weight" % layer_id] = layer.get_weights()[0]
    weight_dict["fc_%d_bias" % layer_id] = layer.get_weights()[1]
    if len(layers_list) == 0:  # The first Layer
        input_buf = template.host.buffer(layer.get_input_shape_at(0)[1], "fc_%d_in" % layer_id)
    else:
        input_buf = layers_list[-1].output
    output_buf = template.host.buffer(layer.get_output_shape_at(0)[1], "fc_%d_out" % layer_id)

    layers_list.append(template.dense.dense(input_buf, output_buf,
                                            "fc_%d" % layer_id, "fc_%d_weight" % layer_id, "fc_%d_bias" % layer_id))

    return weight_dict, layers_list

def _add_bn(layer, weight_dict, layers_list, layer_id):
    gamma = layer.get_weights()[0]
    beta = layer.get_weights()[1]
    moving_mean = layer.get_weights()[2]
    moving_var = layer.get_weights()[3]

    par_a = gamma / np.sqrt(moving_var ** 2 + layer.epsilon)
    par_b = moving_mean / np.sqrt(moving_var ** 2 + layer.epsilon) + beta

    weight_dict["bn_%d_par_a" % layer_id] = par_a
    weight_dict["bn_%d_par_b" % layer_id] = par_b

    input_buf = layers_list[-1].output
    output_buf = template.host.buffer(layer.get_output_shape_at(0)[1], "bn_%d_out" % layer_id)

    layers_list.append(template.normalization.batch_norm(input_buf, output_buf,
                                            "bn_%d" % layer_id, "bn_%d_par_a" % layer_id, "bn_%d_par_b" % layer_id))

    return weight_dict, layers_list

def _add_conv(layer, weight_dict, layers_list, layer_id):
    weight_dict["conv_%d_weight" % layer_id] = layer.get_weights()[0]
    weight_dict["conv_%d_bias" % layer_id] = layer.get_weights()[1]
    if len(layers_list) == 0:  # The first Layer
        input_buf = template.host.buffer(layer.get_input_shape_at(0)[1:], "conv_%d_in" % layer_id)
    else:
        input_buf = layers_list[-1].output
    output_buf = template.host.buffer(layer.get_output_shape_at(0)[1:], "conv_%d_out" % layer_id)

    conf = layer.get_config()

    if isinstance(layer, layers.Conv2D):
        layers_list.append(template.conv.conv2D(input_buf, output_buf,
                                            "conv_%d" % layer_id, "conv_%d_weight" % layer_id, "conv_%d_bias" % layer_id,
                                                stride=conf['strides'], padding = 1 if conf['padding']=='valid' else 0,
                                                num_filter=layer.get_weights()[0].shape[3], filter_size=layer.get_weights()[0].shape[0],
                                                img_size=layer.get_input_shape_at(0)[1], input_channels=layer.get_input_shape_at(0)[3],
                                                output_size=layer.get_output_shape_at(0)[1]))
    elif isinstance(layer, layers.Conv1D):
        layers_list.append(template.conv.conv1D(input_buf, output_buf,
                                                "conv_%d" % layer_id, "conv_%d_weight" % layer_id,
                                                "conv_%d_bias" % layer_id,
                                                stride=conf['strides'], padding=1 if conf['padding'] == 'valid' else 0,
                                                num_filter=layer.get_weights()[0].shape[3],
                                                filter_size=layer.get_weights()[0].shape[0]))
    else:
        print("Not a supported conv operation.")

    return weight_dict, layers_list

def _add_act(layer, layers_list, layer_id, type=None):
    input_buf = layers_list[-1].output
    output_buf = template.host.buffer(input_buf.size, "act_%d_out" % layer_id)
    params = None
    if type is None:
        if isinstance(layer, layers.ReLU) :
            types = 'relu'
        elif isinstance(layer, layers.ThresholdedReLU):
            types = 'threshrelu'
            params = [layer.theta, ]
        elif isinstance(layer, layers.LeakyReLU):
            types = 'leakyrelu'
            params = [layer.alpha, ]
        elif isinstance(layer, layers.PReLU):
            types = 'prelu'
            params = [layer.alpha, ]
        elif isinstance(layer, layers.ELU):
            types = 'elu'
            params = [layer.alpha, ]
        elif isinstance(layer, layers.Softmax):
            types = 'softmax'
        # elif isinstance(layer, keras.activations.sigmoid):
        #     types = 'sigmoid'
    act = template.activation.activation(input_buf, output_buf, "act_%d" % layer_id, types, params)
    layers_list.append(act)

def _add_pool(layer, layers_list, layer_id):
    input_buf = layers_list[-1].output
    output_buf = template.host.buffer(layer.get_output_shape_at(0)[1:], "pool_%d_out" % layer_id)
    conf = layer.get_config()
    params = [conf['pool_size'][0], conf['pool_size'][1],
              layer.get_output_shape_at(0)[1], layer.get_output_shape_at(0)[2],
              conf['strides'][0], conf['strides'][1]]

    act = template.pool.pool(input_buf, output_buf, "pool_%d" % layer_id, params, 'max' if isinstance(layer, layers.MaxPool2D) else 'avg')
    layers_list.append(act)

    return layers_list
