from AIPUBuilder.core.tensor import *
from AIPUBuilder.core.node import *
from AIPUBuilder.core.graph import *
from AIPUBuilder.core.utils import RaiseCRuntimeError, DisableLogger
from AIPUBuilder.logger import get_error_count, INFO
import AIPUBuilder._C._ops as _ops
from AIPUBuilder._C._core import Quantization
from AIPUBuilder.core import run_node
from functools import wraps
import numpy as np
import torch
import sys
import io


_EAGER_MODE = False
_EAGER_MODE_ENGINE = ""


def get_eager_mode():
    return _EAGER_MODE


def get_eager_mode_engine():
    return _EAGER_MODE_ENGINE


def enable_eager_mode(engine=""):
    global _EAGER_MODE
    global _EAGER_MODE_ENGINE
    _engines = {
        "opt": "_opt",
        "sim": "_elf",
        "npu": "_device",
        "exe": "_exe",
    }
    _EAGER_MODE = True
    _EAGER_MODE_ENGINE = _engines.get(engine, "")
    INFO(f"enable eager mode with engine={_EAGER_MODE_ENGINE}")


def disable_eager_mode():
    global _EAGER_MODE
    global _EAGER_MODE_ENGINE
    _EAGER_MODE = False
    _EAGER_MODE_ENGINE = ""


class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def put_raw_tensor_in_graph(t: Tensor, g=None):
    if t.op is not None:
        return
    if g is None:
        g = get_default_graph()
    _ops.input(t, g)


def check_input_op(func):
    """Check all the arguments and add it to graph if one argument
    is a tensor and its op is None.
    """

    def add_tensor_to_graph(arg):
        if isinstance(arg, Tensor):
            if arg.op is None:
                # make the node as input
                g = get_default_graph()
                put_raw_tensor_in_graph(arg, g)

    @wraps(func)
    def wrap(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (list, TensorList)):
                # for the case that input is a list of tensors
                for t in arg:
                    add_tensor_to_graph(t)
            else:
                add_tensor_to_graph(arg)

        for arg in kwargs.values():
            if isinstance(arg, (list, TensorList)):
                # for the case that input is a list of tensors
                for t in arg:
                    add_tensor_to_graph(t)
            else:
                add_tensor_to_graph(arg)

        return func(*args, **kwargs)

    return wrap


def check_input_arg_op(func):
    """Check input in arguments and add it to graph if the
    input is a tensor and its op is None.

    Apply to functions with the first argument as an input
    tensor like (tensor, others).
    It also applies to the functions with the input tensor
    specified by "input" key in arguments like
    (input=tensor, otherarg=others).
    """

    @wraps(func)
    def wrap(*args, **kwargs):
        if len(args) > 0:
            input_arg = args[0]
        elif len(kwargs.keys()) > 0:
            input_arg = kwargs["input"]
        else:
            return func(*args, **kwargs)

        if isinstance(input_arg, Tensor):
            if input_arg.op is None:
                # make the node as input
                g = get_default_graph()
                put_raw_tensor_in_graph(input_arg, g)

        return func(*args, **kwargs)

    return wrap


def check_2_input_arg_op(func):
    """Check input in arguments and add it to graph if the
    input is a tensor and its op is None.

    Apply to functions with the first argument as an input
    tensor like (tensor, others).
    It also applies to the functions with the input tensor
    specified by "scores,coordinates" key in arguments like
    (scores=tensor, coordinates=tensor2,otherarg=others).
    """

    @wraps(func)
    def wrap(*args, **kwargs):
        input_args = []
        if len(args) > 0:
            input_args.append(args[0])
        if len(args) > 1:
            input_args.append(args[1])
        if len(kwargs.keys()) > 0:
            if "scores" in kwargs.keys():
                input_args.append(kwargs["scores"])
            if "coordinates" in kwargs.keys():
                input_args.append(kwargs["coordinates"])
        if len(input_args) == 0:
            return func(*args, **kwargs)
        for input_arg in input_args:
            if isinstance(input_arg, Tensor):
                if input_arg.op is None:
                    # make the node as input
                    g = get_default_graph()
                    put_raw_tensor_in_graph(input_arg, g)

        return func(*args, **kwargs)

    return wrap


def check_input_constant_op(func):
    """Check all the arguments and add it to graph if one argument
    is a tensor and its op is None.
    The first input is used as input node, and the other inputs are used as constant node.
    """

    def add_tensor_to_graph(arg, is_constant=False):
        if isinstance(arg, Tensor):
            if arg.op is None:
                # make the node as input
                g = get_default_graph()
                if is_constant:
                    _ops.constant(arg, arg.quantization, g)
                else:
                    _ops.input(arg, g)

    @wraps(func)
    def wrap(*args, **kwargs):
        for i in range(0, len(args)):
            if i == 0:
                add_tensor_to_graph(args[i])
            else:
                add_tensor_to_graph(args[i], True)
        for key, value in kwargs.items():
            if key == "input":
                add_tensor_to_graph(value)
            else:
                add_tensor_to_graph(value, True)

        return func(*args, **kwargs)

    return wrap


def check_rterrors(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        with RaiseCRuntimeError():
            ret = func(*args, **kwargs)
            if isinstance(ret, Tensor):
                ret.op.infer_shape(True)
            else:
                for t in ret:
                    t.op.infer_shape(True)
            return ret

    return wrap


def wrap_numpy_to_tensor(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        args = [Tensor(a) if isinstance(a, np.ndarray) else a for a in args]
        for key, value in kwargs.items():
            kwargs[key] = Tensor(value) if isinstance(value, np.ndarray) else value
        return func(*args, **kwargs)

    return wrap


def eager_mode(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        if not get_eager_mode():
            return func(*args, **kwargs)
        else:
            with DisableLogger():
                ret = func(*args, **kwargs)
                op = ret[0].op if isinstance(ret, (tuple, list)) else ret.op
                if _EAGER_MODE_ENGINE != "":
                    op.attrs[_EAGER_MODE_ENGINE] = True
                run_node(op)
            return ret

    return wrap


def make_sure_all_tensors(*args):
    t = None
    ret = []
    for i in args:
        if isinstance(i, Tensor):
            t = i
            break
    if t is None:
        raise TypeError(
            "arguments must have at least one tensor, but got 0: %s" % (str(args))
        )
    g = t.op.graph
    for i in args:
        if not isinstance(i, Tensor):
            if isinstance(i, (int, float)):
                i = Tensor(g.get_valid_tensor_name("s"), np.array([i]))
                inp = Node(g.get_valid_node_name("s"), OpType.Constant)
                inp.add_output(i)
                inp.constants["weights"] = i
                g.add_node(inp)
            else:
                raise TypeError("Only accept int/float/Tensor type, but got %s" % (i))
        ret.append(i)
    return tuple(ret)


def convert_activation(act) -> _ops.Activation:
    ret = _ops.Activation("NONE")
    dict = {
        "NONE": _ops.Activation("NONE"),
        "RELU": _ops.Activation("RELU"),
        "RELU1": _ops.Activation("RELU1"),
        "RELU6": _ops.Activation("RELU6"),
        "LEAKYRELU": _ops.Activation("LEAKYRELU"),
        "PRELU": _ops.Activation("PRELU"),
        "ELU": _ops.Activation("ELU"),
        "AFFINE": _ops.Activation("AFFINE"),
        "SIGMOID": _ops.Activation("SIGMOID"),
        "SOFTSIGN": _ops.Activation("SOFTSIGN"),
        "SOFTPLUS": _ops.Activation("SOFTPLUS"),
        "TANH": _ops.Activation("TANH"),
        "THRESHOLDEDRELU": _ops.Activation("THRESHOLDEDRELU"),
        "HARDSIGMOID": _ops.Activation("HARDSIGMOID"),
        "HARDSIGHMOID": _ops.Activation("HARDSIGHMOID"),
        "CLIP": _ops.Activation("CLIP"),
    }
    params_dict = {
        "LEAKYRELU": 1,
        "CLIP": 2,
        "PRELU": 1,
        "HARDSIGHMOID": 2,
    }

    if isinstance(act, list):
        new_act = None
        name = act[0]
        if name.upper() in dict:
            new_act = dict[name.upper()]
            index = 1
            params_f = []
            params_i = []
            while index < len(act):
                if isinstance(act[index], float):
                    params_f.append(act[index])
                if isinstance(act[index], int):
                    params_i.append(act[index])
                if isinstance(act[index], bool):
                    new_act.params_quantized = act[index]
                index += 1
            new_act.params_f = params_f
            new_act.params_i = params_i
            act = new_act

    if act is None:
        ret = _ops.Activation("NONE")
    elif isinstance(act, _ops.Activation):
        if act.type in params_dict.keys() and (
            len(act.params_f) != params_dict[act.type]
            and len(act.params_i) != params_dict[act.type]
            and len(act.tensors) != params_dict[act.type]
        ):
            raise ValueError(
                "Op '%s' expected 'activation' %s has %d params."
                % (
                    sys._getframe().f_back.f_code.co_name,
                    act.type,
                    params_dict[act.type],
                )
            )
        if act.type.upper() == "PRELU":
            act.tensors[0] = convert_to_constant(act.tensors[0], "negative_slope")
        ret = act
    elif isinstance(act, str):
        if act.upper() in params_dict.keys():
            raise ValueError(
                "Op '%s' expected 'activation' %s has params, you can construct like 'ops.Activation(\"CLIP\", -1., 1.)'."
                % (sys._getframe().f_back.f_code.co_name, act)
            )
        elif act.upper() in dict:
            ret = dict[act.upper()]
        else:
            raise ValueError(
                "Op '%s' expected 'activation' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), act)
            )
    else:
        raise TypeError(
            "Op '%s' expected 'activation' argument is Activation(), or str, not '%s'"
            % (sys._getframe().f_back.f_code.co_name, type(act))
        )
    return ret


def convert_quantization(quantization) -> Quantization:
    if quantization is None:
        return Quantization()
    elif isinstance(quantization, Quantization):
        return quantization
    elif isinstance(quantization, (float, int)):
        return Quantization(quantization, 0)
    elif isinstance(quantization, (list, tuple)):
        if len(quantization) == 2:
            return Quantization(*tuple(quantization))
        elif len(quantization) == 1:
            return Quantization(*tuple(quantization), 0)
        else:
            raise TypeError(
                " 'quantization' argument form is (scale,offset) or [scale,offset]"
            )
    else:
        raise TypeError(
            "Expected list or Quantization for 'quantization' argument to "
            "'%s' Op, not %r." % (sys._getframe().f_back.f_code.co_name, quantization)
        )


def convert_to_shape(shape) -> TensorShape:
    if isinstance(shape, TensorShape):
        return shape
    elif isinstance(shape, list):
        return TensorShape(shape)
    elif isinstance(shape, tuple):
        return TensorShape(*shape)
    else:
        raise TypeError(
            "Expected list or TensorShape for 'shape' argument to "
            "'%s' Op, not %r." % (sys._getframe().f_back.f_code.co_name, shape)
        )


def convert_to_padmode(mode) -> _ops.PadMode:
    dict = {
        "CONSTANT": _ops.PadMode.Constant,
        "REFLECT": _ops.PadMode.Reflect,
        "SYMMETRIC": _ops.PadMode.Symmetric,
        "EDGE": _ops.PadMode.Edge,
        "WRAP": _ops.PadMode.Wrap,
    }
    if isinstance(mode, _ops.PadMode):
        return mode
    if isinstance(mode, str):
        if mode.upper() in dict:
            return dict[mode.upper()]
        else:
            raise ValueError(
                "Op '%s' expected 'padmode' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), mode)
            )


def convert_to_gru_direction(mode) -> _ops.GruDirection:
    dict = {
        "forward": _ops.GruDirection.Forward,
        "reverse": _ops.GruDirection.Reverse,
    }
    if isinstance(mode, _ops.GruDirection):
        return mode
    if isinstance(mode, str):
        if mode.lower() in dict:
            return dict[mode.lower()]
        else:
            raise ValueError(
                "Op '%s' expected 'GruDirection' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), mode)
            )


def convert_to_gru_out_sequence(mode) -> _ops.GruOutSequence:
    dict = {
        "H": _ops.GruOutSequence.H,
        "Hn": _ops.GruOutSequence.Hn,
        "H_Hn": _ops.GruOutSequence.H_Hn,
    }
    if isinstance(mode, _ops.GruOutSequence):
        return mode
    if isinstance(mode, str):
        if mode in dict:
            return dict[mode]
        else:
            raise ValueError(
                "Op '%s' expected 'GruOutSequence' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), mode)
            )


def convert_to_lstm_out_sequence(mode) -> _ops.LSTMOutSequence:
    dict = {
        "Y": _ops.LSTMOutSequence.Y,
        "H": _ops.LSTMOutSequence.H,
        "C": _ops.LSTMOutSequence.C,
        "Y_H": _ops.LSTMOutSequence.Y_H,
        "Y_C": _ops.LSTMOutSequence.Y_C,
        "H_C": _ops.LSTMOutSequence.H_C,
        "Y_H_C": _ops.LSTMOutSequence.Y_H_C,
    }
    if isinstance(mode, _ops.LSTMOutSequence):
        return mode
    if isinstance(mode, str):
        if mode in dict:
            return dict[mode]
        else:
            raise ValueError(
                "Op '%s' expected 'LSTMOutSequence' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), mode)
            )


def convert_to_list(s, name, length=2):
    if isinstance(s, (list, tuple)):
        if len(s) != length:
            raise ValueError(
                "the %s only accept len=%d, but got %d" % (name, length, len(s))
            )
        if isinstance(s, tuple):
            return list(s)
        return s
    elif isinstance(s, (int, float)):
        return [s] * length
    else:
        raise TypeError(
            "name %s with len %d only accept int,float,tuple,list type."
            % (name, length)
        )


def convert_to_strides(s, l=2):
    return convert_to_list(s, "strides", l)


def convert_to_pads(s, l=4):
    return convert_to_list(s, "pads", l)


def convert_to_padding(padding, dim=4):
    if isinstance(padding, (list, tuple, int)):
        return _ops.Padding(convert_to_list(padding, "pads", 2 * (dim - 2)))
    else:
        return _ops.Padding(padding)


def convert_to_dilations(s, l=2):
    return convert_to_list(s, "dilations", l)


def convert_to_kernels(s, l=2):
    return convert_to_list(s, "kernels", l)


def convert_to_flattendim(flatten_dim):
    flatten_dim_dict = {
        "HW": _ops.FlattenDim.HW,
        "HWC": _ops.FlattenDim.HWC,
        "NHWC": _ops.FlattenDim.NHWC,
        "NCHW": _ops.FlattenDim.NCHW,
    }
    if isinstance(flatten_dim, _ops.FlattenDim):
        return flatten_dim
    if isinstance(flatten_dim, str):
        if flatten_dim.upper() in flatten_dim_dict:
            return flatten_dim_dict[flatten_dim.upper()]
        else:
            raise ValueError(
                "Op '%s' expected 'flatten_dim' argument is %s, not '%s'"
                % (
                    sys._getframe().f_back.f_code.co_name,
                    flatten_dim_dict.keys(),
                    flatten_dim,
                )
            )


def convert_to_clip_mode(clip_mode):
    dict = {
        "SATURATION": _ops.ClipMode.Saturation,
        "TRUNCATION": _ops.ClipMode.Truncation,
    }
    if isinstance(clip_mode, _ops.ClipMode):
        return clip_mode
    if isinstance(clip_mode, str):
        if clip_mode.upper() in dict:
            return dict[clip_mode.upper()]
        else:
            raise ValueError(
                "Op '%s' expected 'clip_mode' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), clip_mode)
            )


def convert_to_resize_method(method):
    dict = {
        "BILINEAR": _ops.ResizeMethod.Bilinear,
        "NEAREST": _ops.ResizeMethod.NearestNeighbor,
    }
    if isinstance(method, _ops.ResizeMethod):
        return method
    if isinstance(method, str):
        if method.upper() in dict:
            return dict[method.upper()]
        else:
            raise ValueError(
                "Op '%s' expected 'method' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), method)
            )


def convert_to_resize_mode(mode):
    dict = {
        "HALFPIXEL": _ops.ResizeMode.HalfPixel,
        "HALFPIXELSYMMETRIC": _ops.ResizeMode.HalfPixelSymmetric,
        "ALIGNCORNERS": _ops.ResizeMode.AlignCorners,
        "ASYMMETRIC": _ops.ResizeMode.Asymmetric,
        "PYTORCHHALFPIXEL": _ops.ResizeMode.PytorchHalfPixel,
        "TFHALFPIXELFORNN": _ops.ResizeMode.TfHalfPixelForNN,
    }
    if isinstance(mode, _ops.ResizeMode):
        return mode
    if isinstance(mode, str):
        if mode.upper() in dict:
            return dict[mode.upper()]
        else:
            raise ValueError(
                "Op '%s' expected 'mode' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), mode)
            )


def convert_to_resize_nearest_mode(mode):
    dict = {
        "ROUNDPREFERFLOOR": _ops.ResizeNearestMode.RoundPreferFloor,
        "ROUNDPREFERCEIL": _ops.ResizeNearestMode.RoundPreferCeil,
        "FLOOR": _ops.ResizeNearestMode.Floor,
        "CEIL": _ops.ResizeNearestMode.Ceil,
        "SIMPLE": _ops.ResizeNearestMode.Simple,
    }
    if isinstance(mode, _ops.ResizeNearestMode):
        return mode
    if isinstance(mode, str):
        if mode.upper() in dict:
            return dict[mode.upper()]
        else:
            raise ValueError(
                "Op '%s' expected 'mode' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), mode)
            )


def convert_to_nms_method(method):
    dict = {
        "HARD": _ops.NmsMethod.Hard,
        "LINEAR": _ops.NmsMethod.Linear,
        "GAUSSIAN": _ops.NmsMethod.Gaussian,
    }
    if isinstance(method, _ops.NmsMethod):
        return method
    if isinstance(method, str):
        if method.upper() in dict:
            return dict[method.upper()]
        else:
            raise ValueError(
                "Op '%s' expected 'method' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), method)
            )


def convert_to_topk_select_method(method):
    dict = {
        "FIRST": _ops.TopkSelectMethod.First,
        "LAST": _ops.TopkSelectMethod.Last,
        "RANDOM": _ops.TopkSelectMethod.Random,
    }
    if isinstance(method, _ops.TopkSelectMethod):
        return method
    if isinstance(method, str):
        if method.upper() in dict:
            return dict[method.upper()]
        else:
            raise ValueError(
                "Op '%s' expected 'method' argument is %s,not '%s'"
                % (sys._getframe().f_back.f_code.co_name, dict.keys(), method)
            )


def check_axis_value(axis, dim_len):
    if isinstance(axis, int):
        if axis < -dim_len or axis > dim_len - 1:
            raise ValueError(
                "argument 'axis' Accepted range is [-%d, %d]" % (-dim_len, dim_len - 1)
            )
    elif isinstance(axis, (list, tuple)):
        for ax in axis:
            if ax < -dim_len or ax > dim_len - 1:
                raise ValueError(
                    "argument 'axis' Accepted range is [-%d, %d]"
                    % (-dim_len, dim_len - 1)
                )
    else:
        raise TypeError("axis must be list, tuple or int, but now %s" % (type(axis)))

    return


def check_is_constants_tensor(t):
    if isinstance(t, Tensor):
        if t.numpy() is None:
            raise ValueError("The weights must has a constants numpy value.")
        t = t.numpy()
    elif isinstance(t, np.ndarray):
        pass
    # todo add more value type
    else:
        raise ValueError("The weights must accept Tensor,numpy array.")
    return t


def convert_to_constant(tensor, name="constant", size=0):
    if isinstance(tensor, Tensor):
        return tensor

    if isinstance(tensor, np.ndarray):
        d = tensor
    elif isinstance(tensor, torch.Tensor):
        d = tensor.cpu().numpy()
    else:  # tensor is None:
        d = np.zeros(size)

    tensor = Tensor(name, d)

    if tensor.dtype == Dtype.VOID:
        tensor._set_dtype(Dtype.INT32)
    # tensor.quantization.primitive_dtype = tensor.dtype
    return tensor


def check_tensor_shape(tensor, expect_dim=4):
    dim_len = tensor.shape.__len__()
    if isinstance(expect_dim, int):
        if dim_len != expect_dim:
            string_dim = "%d-D " % expect_dim
            raise TypeError(
                "Op '%s' accept %s shape input. Actually, the dimension is:%d."
                % (sys._getframe().f_back.f_code.co_name, string_dim, dim_len)
            )
    elif isinstance(expect_dim, (list, tuple)):
        if dim_len not in expect_dim:
            string_dim = None
            for dim in expect_dim:
                string_dim += "%d-D " % dim
            raise TypeError(
                "Op '%s' accept %s shape input. Actually, the dimension is:%d."
                % (sys._getframe().f_back.f_code.co_name, string_dim, dim_len)
            )


def check_shape_equal(tensor, axis, expected, exp_axis=None):
    shape = tensor.shape
    if len(shape) <= axis:
        raise ValueError(
            "Op '%s'  tensor %s got a shape(%s) which dim is < %d."
            % (
                sys._getframe().f_back.f_code.co_name,
                tensor.name,
                str(tensor.shape),
                axis,
            )
        )
    if isinstance(expected, int):
        if shape[axis] != expected:
            raise ValueError(
                "Op '%s'  tensor %s got a shape(%s)  where shape[%d]=%d NOT equal %d."
                % (
                    sys._getframe().f_back.f_code.co_name,
                    tensor.name,
                    str(shape),
                    axis,
                    shape[axis],
                    expected,
                )
            )
    elif isinstance(expected, (list, tuple, TensorShape)) and exp_axis is not None:
        if shape[axis] != expected[exp_axis]:
            raise ValueError(
                "Op '%s'  tensor %s got a shape(%s)  where shape[%d]=%d NOT equal expected(%s)[%d]=%d."
                % (
                    sys._getframe().f_back.f_code.co_name,
                    tensor.name,
                    str(shape),
                    axis,
                    shape[axis],
                    str(expected),
                    exp_axis,
                    expected[exp_axis],
                )
            )
    elif isinstance(expected, (Tensor)) and exp_axis is not None:
        if shape[axis] != expected.shape[exp_axis]:
            raise ValueError(
                "Op '%s'  tensor %s got a shape(%s)  where shape[%d]=%d NOT equal Tensor %s's shape(%s)[%d]=%d."
                % (
                    sys._getframe().f_back.f_code.co_name,
                    tensor.name,
                    str(shape),
                    axis,
                    shape[axis],
                    expected.name,
                    str(expected.shape),
                    exp_axis,
                    expected.shape[exp_axis],
                )
            )
    else:
        raise ValueError(
            "Op '%s'  got wrong parameter checking."
            % (sys._getframe().f_back.f_code.co_name)
        )


def is_unibroadcast(tensor1, tensor2) -> bool:
    dims1 = list(tensor1.shape)
    dims2 = list(tensor2.shape)
    diff_len = len(dims1) - len(dims2)
    dims2[0:0] = [1] * diff_len
    for i in range(len(dims1)):
        if dims1[i] != dims2[i] and dims2[i] != 1:
            return False
    return True


def str_to_enum_member(
    enum_obj, match_str, ignore_case=True, ignore_str_underline=True
):
    if ignore_str_underline:
        match_str = match_str.replace("_", "")
    if ignore_case:
        enum_members = [
            member
            for name, member in enum_obj.__members__.items()
            if name.upper() == match_str.upper()
        ]
    else:
        enum_members = [
            member for name, member in enum_obj.__members__.items() if name == match_str
        ]
    if not enum_members:
        raise TypeError(
            "Value %s is not supported! Supported values are %s"
            % (match_str, enum_obj.__members__.keys())
        )
    return enum_members[0]


def convert_block_shape(block_shape: int or list, length=2):
    if isinstance(block_shape, int):
        block_shape = [block_shape] * length
    elif isinstance(block_shape, (list, tuple)):
        block_shape = list(block_shape)
        if len(block_shape) > length:
            raise ValueError(
                " argument 'block_shape' must Meet the conditions 'len <= 2', not %d"
                % len(block_shape)
            )
        elif len(block_shape) == 1:
            block_shape = [block_shape[0]] * length
    else:
        raise ValueError(" argument 'block_shape' invalid %s" % block_shape)

    return block_shape


def check_crops_or_padding(cps, length=2):
    if not isinstance(cps, (list, tuple)):
        raise TypeError(
            " argument 'crops' or 'paddings' must have 2-D shape `[%d, 2]` , not %r "
            % (length, cps)
        )

    if len(cps) != length:
        raise TypeError(
            " argument 'crops' or 'paddings' must have 2-D shape `[%d, 2]` , not %r "
            % (length, cps)
        )

    for cp in cps:
        if not isinstance(cp, (list, tuple)):
            raise TypeError(
                " argument 'crops' or 'paddings' must be list or tuple , not %r "
                % (type(cp))
            )
        if len(cp) != 2:
            raise ValueError(
                " argument 'crops' or 'paddings' must be [start, end] , not %r " % cp
            )
    return


def check_same_shape(input1: Tensor, input2: Tensor):
    if input1.shape != input2.shape:
        raise ValueError(
            "Op 'logical_and' expected argument 'input1' and argument 'input2' have the same shape."
        )
    return


def confirm_data_layout(tensor: Tensor, layout=DataLayout.NHWC):
    if tensor.shape.layout == DataLayout.Flat:
        tensor.shape.layout = layout
    if tensor.shape.layout != layout:
        raise TypeError(
            "'%s' layout Only Support %r, not %r."
            % (sys._getframe().f_back.f_code.co_name, layout, tensor.shape.layout)
        )


def confirm_tensor_dtype(tensor: Tensor, tensors2: Tensor, dtype_matrix: dict):
    if tensor.dtype not in dtype_matrix:
        return
        raise TypeError(f"{tensor.name} only accepts dtypes: {dtype_matrix.keys()}.")
    if tensors2.dtype not in dtype_matrix[tensor.dtype]:
        raise TypeError(
            f"{tensors2.name} only accepts dtypes: {dtype_matrix[tensor.dtype]}."
        )
