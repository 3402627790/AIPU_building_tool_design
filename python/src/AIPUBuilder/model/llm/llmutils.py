from AIPUBuilder.logger import WARN, DEBUG, ERROR, INFO
from AIPUBuilder.model.register import CAUSALLMS
from AIPUBuilder.core import OpType


def get_embedding_data(config, model_type, force_float=True):
    if model_type.endswith("_embedding"):
        DEBUG(f"{model_type} is embedding data, so will skip get embedding data process.")
        return None
    embedding_model_name = f'{model_type}_embedding'
    if embedding_model_name not in CAUSALLMS:
        WARN(f"{embedding_model_name} not in {CAUSALLMS.keys()} when get embedding data.")
        return None
    else:
        return CAUSALLMS[embedding_model_name](config).embedding_data(force_float)


def build_ir_postprocess(model, config):

    if config.get('hf_quantization_config', None) is not None:
        import torch
        hqc = config.hf_quantization_config
        quant_method = hqc.get('quant_method', None)
        if quant_method == 'compressed-tensors':
            # config_groups = hqc.get("config_groups")

            for n in model.nodes:
                if n.type == OpType.FullyConnected:
                    w = n.constants['weights']
                    if 'input_scale' in w._attrs:
                        input_scale = w._attrs['input_scale']
                        input_t = n.inputs[0]
                        input_t.quantization.scales = input_scale
                        input_t.quantization.offsets = torch.zeros_like(input_scale)
                        n.params['input_scale'] = input_scale.item()
                    elif 'input_global_scale' in w._attrs:
                        input_scale = w._attrs['input_global_scale']
                        input_t = n.inputs[0]
                        input_t.quantization.scales = input_scale
                        input_t.quantization.offsets = torch.zeros_like(input_scale)
                        n.params['input_global_scale'] = input_scale.item()
                    # elif 'weight_global_scale' in w._attrs:
                    #     weight_global_scale = w._attrs['weight_global_scale']
                    #     n.params['weight_global_scale'] = weight_global_scale.item()


def gen_layer_ids(num_layers):
    if not isinstance(num_layers, (int, list)):
        ERROR(f"gen_layer_ids only supports integer or list, now its type is {type(num_layers)}.")

    ret = []
    if isinstance(num_layers, int):
        DEBUG(f"now num_hidden_layers = {num_layers}")
        return list(range(num_layers))

    if isinstance(num_layers, list):
        ret = []
        if isinstance(num_layers[0], list):
            if not all(type(nl) is list for nl in num_layers):
                ERROR(
                    f"num_hidden_layers only supports nested list, like num_hidden_layers=[[2, 2], [3,7]] format, or single list or integer. now is {num_layers}")
        else:
            if not all(type(nl) is int for nl in num_layers):
                ERROR(
                    f"num_hidden_layers only supports single list, like num_hidden_layers=[2, 4] or nested list, like num_hidden_layers=[[2, 2], [3,7]] format. now is {num_layers}")

        if isinstance(num_layers[0], list):
            for nl in num_layers:
                ret.extend(list(range(nl[0], nl[1] + 1)))
        else:
            ret.extend(list(range(num_layers[0], num_layers[1] + 1)))
        DEBUG(f"now num_hidden_layers list = {ret}")
        return ret
