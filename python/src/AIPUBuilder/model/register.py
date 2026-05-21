from AIPUBuilder.logger import INFO, DEBUG, ERROR
from AIPUBuilder.core import Graph, Tensor, Node, TensorList, Dtype, OpType
from AIPUBuilder import ops
from AIPUBuilder import nn

CAUSALLMS = {}
MODELQUANTIZE = {}
MODELBUILD = {}
APP = {}
ENGINES = {}
CHATTEMPLATE = {}
UPDATECONFIG = {}
GGUFUNPACK = {}
QUANTMETHODQTRANS = {}
QUANTMETHODS = {}
TORCHVISIONHUB = {}


def _register(names, registed_map_key):
    registed_map = globals()[registed_map_key]
    if isinstance(names, (str)):
        names = [names, ]

    def wrapper(func):
        for name in names:
            if name.lower() not in registed_map:
                registed_map.update({name.lower(): func})
            else:
                INFO(f"{name} has already registered.")
        return func
    return wrapper


def register_torchvision_hub(names):
    return _register(names, 'TORCHVISIONHUB')


def register_quant_method(names):
    return _register(names, 'QUANTMETHODS')


def register_quant_method_trans(names):
    return _register(names, 'QUANTMETHODQTRANS')


# def register_quant_method_trans(names):
#     if isinstance(names, (str)):
#         names = [names, ]

#     def wrapper(func):
#         for name in names:
#             if name.lower() not in QUANTMETHODQTRANS:
#                 QUANTMETHODQTRANS.update({name.lower(): func})
#             else:
#                 INFO(f"{name} quant_method has already registered in QUANTMETHODQTRANS")
#         return func
#     return wrapper


def register_causallm(names):
    if not isinstance(names, (list, tuple)):
        names = list(names)

    def wrapper(func):
        for name in names:
            if name.lower() not in CAUSALLMS:
                from AIPUBuilder.model.basemodel import BaseEmbedding
                if issubclass(func, BaseEmbedding):
                    if not name.endswith('_embedding'):
                        ERROR(f"[{name}] for registering embedding IR should end with '_embedding'.")
                CAUSALLMS.update({name.lower(): func})
            else:
                INFO(f"{name} causllm has already registered in CAUSALLMS")
        return func
    return wrapper


register_nn = register_causallm


def register_model_quantize(names):
    if isinstance(names, (str)):
        names = [names, ]

    def wrapper(func):
        for name in names:
            if name.lower() not in MODELQUANTIZE:
                MODELQUANTIZE.update({name.lower(): func})
            else:
                INFO(f"{name} causllm has already registered in MODELQUANTIZE")
        return func
    return wrapper


def register_model_build(names):
    if isinstance(names, (str)):
        names = [names, ]

    def wrapper(func):
        for name in names:
            if name.lower() not in MODELBUILD:
                MODELBUILD.update({name.lower(): func})
            else:
                INFO(f"{name} causllm has already registered in MODELBUILD")
        return func
    return wrapper


def register_app():
    def wrapper(cls):
        cls_name = cls.__name__
        if cls_name not in APP:
            APP[cls_name.lower()] = cls
        else:
            INFO(f"{cls_name} app has alread registered in APP")
        return cls
    return wrapper


def register_engine(name):
    def wrapper(cls):
        if name.lower() not in ENGINES:
            ENGINES[name.lower()] = cls
        else:
            INFO(f"{name} app has alread registered in ENGINES")
        return cls
    return wrapper


def register_chat_template(names):
    if isinstance(names, (str)):
        names = [names, ]

    def wrapper(func):
        for name in names:
            if name.lower() not in CHATTEMPLATE:
                CHATTEMPLATE[name.lower()] = func
            else:
                INFO(f"{name} app has alread registered in CHATTEMPLATE")
        return func
    return wrapper


def register_update_config(names):
    if isinstance(names, (str)):
        names = [names, ]

    def wrapper(func):
        for name in names:
            if name.lower() not in UPDATECONFIG:
                UPDATECONFIG[name.lower()] = func
            else:
                INFO(f"{name} app has alread registered in UPDATECONFIG")
        return func
    return wrapper


def register_gguf_unpack(names):
    if not isinstance(names, (list, tuple)):
        names = [names, ]

    def wrapper(func):
        for name in names:
            if name not in GGUFUNPACK:
                GGUFUNPACK.update({name: func})
            else:
                INFO(f"{name} gguf quant-type has already registered in GGUFUNPACK")
        return func
    return wrapper


def gen_subgraph(subgraph_name=None):
    def wrapper(func):
        def label_func(*args, **kwargs):
            m = args[0]
            is_split = False
            if isinstance(m, nn.Module):
                from AIPUBuilder.model.config import PretrainedConfig
                for _, v in m.__dict__.items():
                    if isinstance(v, PretrainedConfig):
                        if hasattr(v, 'enable_subgraph') and v.get('enable_subgraph', False):
                            is_split = True
                            break
            if is_split:
                return _label_subgraph(func, subgraph_name, *args, **kwargs)
            return func(*args, **kwargs)
        return label_func
    return wrapper


def _label_subgraph(func, subgraph_name, *args, **kwargs):
    def _get_cur_graph(iargs):
        g = None
        for ia in iargs:
            if isinstance(ia, Tensor):
                g = ia.op.graph
                break
            elif isinstance(ia, Node):
                g = ia.graph
                break
        return g

    def _get_input_tensors(args):
        subg_in_tensors = []
        for arg in args:
            if isinstance(arg, Tensor):
                subg_in_tensors.append(arg)
        return subg_in_tensors

    all_args = [*args] + [kwargs.values()]
    cur_g = _get_cur_graph(all_args)
    if cur_g is None:
        return func(*args, **kwargs)

    subg_in_tensors = _get_input_tensors(all_args)
    prev_node_names = [n.name for n in cur_g.nodes]
    ret_tensors = func(*args, **kwargs)
    m = args[0]
    model_type = m.config.model_type if hasattr(m.config, 'model_type') else f"_default_{len(cur_g.nodes)}"
    ir_prefix = m.config.ir_prefix if hasattr(m.config, 'ir_prefix') else ""
    output_dir = m.config.get('output_dir', "") + '/'
    ir_prefix = '_' + ir_prefix if len(ir_prefix) else ir_prefix
    lid = m.layer_id if hasattr(m, 'layer_id') else ""
    subgraph_name = subgraph_name if subgraph_name is not None else ""
    call_subgraph = f"{output_dir}{model_type}{ir_prefix}_{subgraph_name}_{lid}"
    postfix = ""

    subg_tensors = []
    for n in cur_g.nodes:
        if n.name not in prev_node_names:
            n.attrs['_is_call'] = call_subgraph
            n.params['_is_call'] = call_subgraph
            subg_tensors.append(n.outputs[0])

    for idx, subg_in_t in enumerate(subg_in_tensors):
        subg_in_t._attrs['_ordered_input_id'] = idx
        subg_in_t._attrs['remove_repeat_tensor'] = True

    rts = ret_tensors
    if not isinstance(ret_tensors, (tuple, list)):
        rts = [rts]
    for idx, rt in enumerate(rts):
        if isinstance(rt, Tensor):
            rt.io_tensor = True
            rt._attrs['_ordered_output_id'] = idx

    call_t = ops.call(subg_tensors, call_subgraph, f"{call_subgraph}{postfix}.txt")
    if 'is_call' not in call_t[0].op.attrs:
        call_t[0].op.attrs['is_call'] = True
    DEBUG(f"[LLM] creat call op: {call_t[0].op}, and its type: {call_t[0].op.type}")
    return ret_tensors
