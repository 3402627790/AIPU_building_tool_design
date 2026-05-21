#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3
from .utils import continue_class, wrap_class_method
import torch
from AIPUBuilder._C._tongue import Expr, Symbol
from AIPUBuilder._C._core import (Tensor,
                                  ComputeTensor,
                                  OpType,
                                  Node,
                                  Dtype,
                                  DataLayout,
                                  TensorList,
                                  TensorShape,
                                  SymbolShape,
                                  _py_register_optype,
                                  _py_register_op_mem_remap)
__all__ = [
    "Tensor",
    "ComputeTensor",
    "TensorShape",
    "SymbolShape",
    "Expr",
    "Symbol",
    "OpType",
    "Dtype",
    "DataLayout",
    "TensorList",
    "register_optype",
    "register_op_mem_remap",
    "use_cuda",
    "using_cuda",
]


def register_optype(t: str):
    '''
    Register a new OpType with a str it will return a int number for the OpType ID

    Args:
       t(str): the name of the op type

    Returns:
         OpType: the id of the op type registered
    '''
    if t not in OpType.__entries:
        v = OpType(_py_register_optype(t))
        setattr(OpType, t, OpType(v))
        OpType.__entries[t] = (v, None)  # (value,doc string)
        return v
    else:
        return OpType.__entries[t][0]


def register_op_mem_remap(t: str):
    '''
    Register an OpType for its mem remapping function

    Args:
        t(str): the name of the op type

    '''
    def decorator(func):
        if isinstance(t, OpType):
            op_names = [
                op_name for op_name in OpType.__entries if OpType.__entries[op_name][0] == t]
            if len(op_names) == 0:
                raise RuntimeError(f"OpType {t} has not been registered")
            op_id, op_name = t, op_names[0]
        elif isinstance(t, str):
            if t not in OpType.__entries:
                raise RuntimeError(f"OpType {t} has not been registered")
            op_id, op_name = OpType.__entries[t][0], t
        else:
            raise RuntimeError(f"OpType {t} should be OpType or str")
        _py_register_op_mem_remap(op_name, func)
        return func
    return decorator


@continue_class
class Dtype:
    import numpy as np
    import torch
    from ml_dtypes import bfloat16, float8_e5m2, float8_e4m3fn, float4_e2m1fn
    _th_dict = {Dtype.BOOL: torch.bool,
                # Dtype.FP4E2M1FN: torch.float4_e2m1fn,
                Dtype.FP8E5M2: torch.float8_e5m2,
                Dtype.FP8E4M3FN: torch.float8_e4m3fn,
                Dtype.FP16: torch.float16,
                Dtype.BFP16: torch.bfloat16,
                Dtype.FP32: torch.float32,
                Dtype.FP64: torch.float64,
                Dtype.INT16: torch.int16,
                Dtype.INT32: torch.int32,
                Dtype.INT64: torch.int64,
                Dtype.INT8: torch.int8,
                Dtype.UINT8: torch.uint8,
                Dtype.VOID: torch.uint8,
                Dtype.UINT16: torch.int32,
                Dtype.UINT32: torch.int64,
                Dtype.ALIGNED_INT4: torch.int8,
                Dtype.ALIGNED_UINT4: torch.uint8,
                Dtype.ALIGNED_INT12: torch.int16,
                Dtype.ALIGNED_UINT12: torch.int16,

                }
    _np_dict = {Dtype.BOOL: bool,
                Dtype.FP4E2M1FN: float4_e2m1fn,
                Dtype.FP8E5M2: float8_e5m2,
                Dtype.FP8E4M3FN: float8_e4m3fn,
                Dtype.FP16: np.float16,
                Dtype.BFP16: bfloat16,
                Dtype.FP32: np.float32,
                Dtype.FP64: np.float64,
                Dtype.INT16: np.int16,
                Dtype.UINT16: np.uint16,
                Dtype.INT32: np.int32,
                Dtype.UINT32: np.uint32,
                Dtype.INT64: np.int64,
                Dtype.INT8: np.int8,
                Dtype.UINT8: np.uint8,
                Dtype.VOID: np.uint8,
                Dtype.ALIGNED_INT4: np.int8,
                Dtype.ALIGNED_UINT4: np.uint8,
                Dtype.ALIGNED_INT12: np.int16,
                Dtype.ALIGNED_UINT12: np.uint16,
                }

    @staticmethod
    def get(dtype):
        """
        Get the corresponding enum value from a given data type.

        Args:
           dtype (str or torch.dtype or numpy.dtype): The data type to convert.

        Returns:
            Dtype: The enum value corresponding to the data type.
        """
        if isinstance(dtype, str):
            for i, v in Dtype.__dict__["__entries"].items():
                if i.lower() == dtype:
                    return v[0]
        if isinstance(dtype, torch.dtype):
            for k, v in Dtype._th_dict.items():
                if v == dtype:
                    return k
        import numpy as np
        dtype = np.dtype(dtype)
        for k, v in Dtype._np_dict.items():
            if v == dtype:
                return k
        return Dtype()

    @staticmethod
    def from_np(dtype):
        """
        Create a Dtype from a numpy data type.

        Args:
            dtype (numpy.dtype): The numpy data type to convert.

        Returns:
            Dtype: The enum value corresponding to the numpy data type.
        """
        import numpy as np
        dtype = np.dtype(dtype)
        for k, v in Dtype._np_dict.items():
            if v == dtype:
                return k
        return Dtype()

    @staticmethod
    def from_torch(dtype):
        """
        Create a Dtype from a PyTorch data type.

        Args:
            dtype (torch.dtype): The PyTorch data type to convert.

        Returns:
            Dtype: The enum value corresponding to the PyTorch data type.
        """
        if isinstance(dtype, torch.dtype):
            for k, v in Dtype._th_dict.items():
                if v == dtype:
                    return k
        return Dtype()

    @property
    def np(self):
        """
        Return the numpy data type corresponding to this enum value.

        Returns:
            numpy.dtype: The numpy data type corresponding to this enum value.
        """
        return self._np_dict[self]

    @property
    def torch(self):
        """
        Return the PyTorch data type corresponding to this enum value.

        Returns:
            torch.dtype: The PyTorch data type corresponding to this enum value.
        """
        return self._th_dict[self]

    def __str__(self) -> str:
        return self.__to_str__()


TensorList.__repr__ = TensorList.__str__


@continue_class
class TensorShape:
    def clone(self):
        """
        Clone this TensorShape.

        Returns:
            TensorShape: A new TensorShape with the same values.
        """
        return self.__class__(list(self))

    def reshape(self, *args):
        """
        Reshape this TensorShape.

        Args:
           *args: The new shape.

        Returns:
           TensorShape: A new TensorShape with the new shape.
        """
        other = TensorShape(*args)
        if -1 in other:
            s = 1
            idx = -1
            for k, i in enumerate(other):
                if i != -1:
                    s *= i
                elif idx == -1:
                    idx = k
                else:
                    raise ValueError(
                        "Shape only accepts one placeholder as -1.")
            other[idx] = int(self.size() // s)
        if other.size() != self.size():
            raise ValueError("Reshape size %d!=%d mismatch, try to reshape from %s to %s" % (self.size(), other.size(),
                                                                                             str(self), str(other)))
        self.__reshape__(other)
        return self

    def index(self, i):
        """
        Get the index of a value in this TensorShape.

        Args:
            i: The value to find.

        Returns:
           int: The index of the value. Raises IndexError if not found.
        """
        for k, j in enumerate(self):
            if j == i:
                return k
        raise IndexError("Can not find the value ", i, "in the Shape.")


_USE_CUDA = False
if torch.cuda.is_available():
    _USE_CUDA = True
    pass


class PythonObjectWrapper(object):
    def __init__(self, t):
        self.obj = t


def use_cuda():
    """
    Check whether CUDA is available for backend tensor computation.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return _USE_CUDA


def _get_torch_tensor(self):
    import torch
    import numpy as np
    if "_betensor_cpu" in self._attrs:
        self.betensor = self._attrs["_betensor_cpu"].obj
        obj = self._attrs.pop("_betensor_cpu")

        obj.obj = None
    if "_betensor" not in self._attrs:
        arr = self.numpy()
        if arr is None:
            arr = np.zeros(list(self.shape)).astype(self.dtype.np)
        self.betensor = arr
    return self._attrs["_betensor"].obj


def _set_torch_tensor(self, t):
    import torch
    import numpy as np
    from ml_dtypes import bfloat16, float8_e5m2, float8_e4m3fn, float4_e2m1fn
    # if t is self._attrs.get("_betensor",PythonObjectWrapper(None)).obj:
    #     return
    if not isinstance(t, torch.Tensor):
        try:
            # pytorch currently accepts ndarray with dtypes of numpy.float64, numpy.float32, numpy.float16, numpy.complex64, numpy.complex128, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint8, and numpy.bool.
            if t.dtype == np.uint16:
                t = t.astype(np.int32)
            elif t.dtype == np.uint32:
                t = t.astype(np.int64)
            elif t.dtype == np.uint64:
                t = t.astype(np.float64)
            elif t.dtype == bfloat16:
                t = t.astype(np.float32)
            elif t.dtype == float8_e5m2:
                t = t.astype(np.float32)
            elif t.dtype == float8_e4m3fn:
                t = t.astype(np.float32)
            elif t.dtype == float4_e2m1fn:
                t = t.astype(np.float32)
            else:
                pass
            t = torch.from_numpy(t)
            # t = torch.from_numpy(t.astype(np.float32)) #.type(self.betensor_type)
        except Exception:
            raise TypeError(
                "betensor must be a torch tensor but get a type %s" % (type(t)))
    try:
        if self.numpy() is None:  # when the data is empty, copy the data to C++
            data = t.float() if self.dtype.np in [bfloat16, float8_e5m2, float8_e4m3fn] else t
            self.set_data(data.numpy().astype(self.dtype.np))
    except Exception as e:
        pass
    if self.dtype.np == np.float16 and t.dtype != torch.float16:
        t = t.half()  # fp32 -> fp16, align to graph
    if self.dtype.np == bfloat16 and t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    if self.dtype.np == float8_e5m2 and t.dtype != torch.float8_e5m2:
        t = t.to(torch.float8_e5m2)
    if self.dtype.np == float8_e4m3fn and t.dtype != torch.float8_e4m3fn:
        t = t.to(torch.float8_e4m3fn)
    self._attrs["_betensor"] = PythonObjectWrapper(t)


def _get_torch_tensor_cuda(self):
    import torch
    import numpy as np
    if "_betensor" not in self._attrs:
        arr = self.numpy()
        if arr is None:
            arr = np.zeros(list(self.shape))
        # pytorch currently accepts ndarray with dtypes of numpy.float64, numpy.float32, numpy.float16, numpy.complex64, numpy.complex128, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint8, and numpy.bool.
        va = torch.from_numpy(arr.astype(np.float32)).cuda()
        self._attrs["_betensor"] = PythonObjectWrapper(va)
    return self._attrs["_betensor"].obj


def _set_torch_tensor_cuda(self, t):
    import torch
    import numpy as np
    from ml_dtypes import bfloat16, float8_e5m2, float8_e4m3fn
    if not isinstance(t, torch.Tensor):
        try:
            # pytorch currently accepts ndarray with dtypes of numpy.float64, numpy.float32, numpy.float16, numpy.complex64, numpy.complex128, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint8, and numpy.bool.
            if t.dtype == np.uint16:
                t = t.astype(np.int32)
            elif t.dtype == np.uint32:
                t = t.astype(np.int64)
            elif t.dtype in [np.uint64, np.float64]:
                t = t.astype(np.float32)
            elif t.dtype == bfloat16:
                t = t.astype(np.float32)
            elif t.dtype == float8_e5m2:
                t = t.astype(np.float32)
            elif t.dtype == float8_e4m3fn:
                t = t.astype(np.float32)
            else:
                pass
            # t = torch.from_numpy(t.astype(np.float32)).cuda()
            t = torch.from_numpy(t).cuda()
        except Exception:
            raise TypeError(
                "betensor must be a torch tensor but get a type %s" % (type(t)))
    elif not t.is_cuda:
        t = t.cuda()
    try:
        if self.numpy() is None:  # when the data is empty, copy the data to C++
            data = t.float() if self.dtype.np in [bfloat16, float8_e5m2, float8_e4m3fn] else t
            self.set_data(data.cpu().numpy().astype(self.dtype.np))
    except Exception as e:
        pass
    if self.dtype.np == np.float16 and t.dtype != torch.float16:
        t = t.half()  # fp32 -> fp16, align to graph
    if self.dtype.np == bfloat16 and t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    if self.dtype.np == float8_e5m2 and t.dtype != torch.float8_e5m2:
        t = t.to(torch.float8_e5m2)
    if self.dtype.np == float8_e4m3fn and t.dtype != torch.float8_e4m3fn:
        t = t.to(torch.float8_e4m3fn)
    self._attrs["_betensor"] = PythonObjectWrapper(t)


def using_cuda(cuda):
    global _USE_CUDA
    _USE_CUDA = cuda
    if not _USE_CUDA:
        Tensor.betensor = property(_get_torch_tensor, _set_torch_tensor)
    else:
        Tensor.betensor = property(
            _get_torch_tensor, _set_torch_tensor_cuda)


using_cuda(_USE_CUDA)


@continue_class
class Tensor:

    @staticmethod
    def new(v):
        import numpy as np
        import torch
        int_size = {
            1: Dtype.INT8,
            2: Dtype.INT16,
            4: Dtype.INT32,
            8: Dtype.INT64,
        }
        dt = Dtype.get(v.dtype)
        if isinstance(v, np.ndarray):
            return Tensor(v.view(int_size[dt.size].np)).view(dt)
        elif isinstance(v, torch.Tensor):
            return Tensor(v.view(int_size[dt.size].torch).cpu().numpy()).view(dt)
        raise TypeError(f"Unsupported type {type(v)}, only support torch tensor and numpy tensor")

    def remove_betensor(self):
        """
        Remove the betensor attribute from the tensor.

        """
        if "_betensor" in self._attrs:
            t = self._attrs.pop("_betensor")
            t.obj = None
            # if use_cuda():
            #     torch.cuda.empty_cache()

    def sync_betensor(self, be2np=True):
        """
        Synchronize the betensor attribute with the tensor data.

        Args:
            be2np (bool): If True, convert the betensor to a numpy array and set it as the tensor data.

        Returns:
            bool: True if synchronization was successful, False otherwise.
        """
        if be2np:
            if "_betensor" in self._attrs:
                try:
                    self.set_data(self.betensor.detach().cpu().view(torch.int8).numpy())
                    return True
                except Exception:
                    self._attrs["_betensor_cpu"] = PythonObjectWrapper(self.betensor.detach().cpu())
                    return True
            return False
        else:
            self.remove_betensor()
            return True

    @property
    def betensor_type(self):
        return self.dtype.torch

    @property
    def dtype(self):
        return self._dtype()

    @dtype.setter
    def dtype(self, dtype):
        import torch
        import numpy as np
        from ml_dtypes import bfloat16, float8_e5m2, float8_e4m3fn
        th_dict = {
            torch.bool: Dtype.BOOL,
            torch.float8_e5m2: Dtype.FP8E5M2,
            torch.float8_e4m3fn: Dtype.FP8E4M3FN,
            torch.float16: Dtype.FP16,
            torch.bfloat16: Dtype.BFP16,
            torch.float32: Dtype.FP32,
            torch.float64: Dtype.FP64,
            torch.int16: Dtype.INT16,
            torch.int32: Dtype.INT32,
            torch.int64: Dtype.INT64,
            torch.int8: Dtype.INT8,
            torch.uint8: Dtype.UINT8,
            torch.uint8: Dtype.VOID,
        }
        np_dict = {
            bool: Dtype.BOOL,
            float8_e5m2: Dtype.FP8E5M2,
            float8_e4m3fn: Dtype.FP8E4M3FN,
            np.float16: Dtype.FP16,
            bfloat16: Dtype.BFP16,
            np.float32: Dtype.FP32,
            np.float64: Dtype.FP64,
            np.int16: Dtype.INT16,
            np.int32: Dtype.INT32,
            np.int64: Dtype.INT64,
            np.int8: Dtype.INT8,
            np.uint8: Dtype.UINT8,
            np.uint8: Dtype.VOID,
        }
        if dtype in th_dict:
            self._set_dtype(th_dict[dtype])
        elif dtype in np_dict:
            self._set_dtype(np_dict[dtype])
        else:
            self._set_dtype(dtype)

    def view(self, dtype):
        """
        view the tensor with a new dtype

        Returns:
            Tensor: a new tensor with the specified dtype
        """
        self.dtype = dtype
        return self

    def numpy(self):
        """
        Get the tensor data as a numpy array

        Returns:
           np.array: the tensor data as a numpy array
        """
        try:
            arr = self.data()
        except (TypeError, ValueError):
            dt = self.dtype
            try:
                if dt.size == 2:
                    self.dtype = Dtype.INT16
                    arr = self.data().view(dt.np)
                elif dt.size == 1:
                    self.dtype = Dtype.INT8
                    arr = self.data().view(dt.np)
                else:
                    raise ValueError("Unsupported data type size")
            except Exception as e:
                self.dtype = dt
                raise e
            self.dtype = dt
        except Exception as e:
            raise e
        return arr

    @property
    def np(self):
        """
        Get the tensor data as a numpy array

        Returns:
            np.array: the tensor data as a numpy array
        """
        return self.numpy()

    @np.setter
    def np(self, v):
        """
        Set the tensor data from a numpy array

        Args:
           v (np.array): the numpy array to set the tensor data from
        """
        self.set_numpy(v)

    @property
    def irtensor(self):
        """
        Get the tensor data same shape and type as in IR

        Returns:
            Tensor: the tensor data same shape and type as in IR
        """
        from .layout_tensfer import non2norm
        layout = self.shape.layout
        func = non2norm.get(layout, lambda x: None)
        return func(self.betensor)

    @irtensor.setter
    def irtensor(self, v):
        """
        Set the tensor data from a IR Tensor

        Args:
            v (Tensor): the IR Tensor to set the tensor data from

        """
        import numpy as np
        from ml_dtypes import bfloat16, float8_e5m2, float8_e4m3fn
        from .layout_tensfer import norm2non
        if isinstance(v, np.ndarray):
            dtype = v.dtype
            if np.issubdtype(v.dtype, np.uint16):
                v = v.astype(np.int32)
            if np.issubdtype(v.dtype, bfloat16):
                v = v.astype(np.float32)
            elif np.issubdtype(v.dtype, float8_e5m2):
                v = v.astype(np.float32)
            elif np.issubdtype(v.dtype, float8_e4m3fn):
                v = v.astype(np.float32)
            v = torch.from_numpy(v)
            if dtype == bfloat16:
                v = v.to(torch.bfloat16)
            elif dtype == float8_e5m2:
                v = v.to(torch.float8_e5m2)
            elif dtype == float8_e4m3fn:
                v = v.to(torch.float8_e4m3fn)
        layout = self.shape.layout
        func = norm2non.get(layout, lambda x: None)
        self.betensor = func(v)

    def clone(self, name):
        """
        Clone the tensor with a new name.

        Args:
           name (str): the new name of the tensor

        Returns:
          Tensor: the cloned tensor with a new name.
        """
        import copy
        tname = name
        t = self.__class__(tname, self.shape, self.dtype)
        t.__dict__.update(copy.deepcopy(self.__dict__))
        for k, v in self._attrs.items():
            try:
                tv = copy.deepcopy(v)
            except:
                continue
            t._attrs.update({k: tv})
        # sys.stdout = temp_stdout
        t.betensor = self.betensor.clone().detach()
        return t

    def __add__(self, other):
        from AIPUBuilder import ops
        return ops.add(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        from AIPUBuilder import ops
        return ops.sub(self, other)

    def __rsub__(self, other):
        from AIPUBuilder import ops
        return ops.sub(other, self)

    def __or__(self, other):
        from AIPUBuilder import ops
        return ops.logical_or(self, other)

    def __gt__(self, other):
        from AIPUBuilder import ops
        return ops.greater(self, other)

    def __ge__(self, other):
        from AIPUBuilder import ops
        return ops.greater_equal(self, other)

    def __lt__(self, other):
        from AIPUBuilder import ops
        return ops.less(self, other)

    def __le__(self, other):
        from AIPUBuilder import ops
        return ops.less_equal(self, other)

    def __and__(self, other):
        from AIPUBuilder import ops
        return ops.logical_and(self, other)

    def __mul__(self, other):
        from AIPUBuilder import ops
        return ops.mul(self, other)

    def __rmul__(self, other):
        return self * other

    def reshape(self, *args):
        from AIPUBuilder import ops
        if len(args) == 1 and isinstance(args[0], (tuple, list, TensorShape, SymbolShape)):
            largs = args[0]
        else:
            largs = [*args]
        return ops.reshape(self, largs)

    def transpose(self, *args):
        from AIPUBuilder import ops
        if len(args) == 1 and isinstance(args[0], (tuple, list, TensorShape)):
            perm = args[0]
        else:
            perm = [*args]
        return ops.transpose(self, perm)

    def tile(self, *args):
        from AIPUBuilder import ops
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            _tiles = args[0]
        else:
            _tiles = [*args]
        return ops.tile(self, _tiles)

    def unsqueeze(self, dim: int):
        from AIPUBuilder import ops
        from AIPUBuilder.logger import ERROR
        dim_range = [-self.shape.dim() - 1, self.shape.dim() + 1]
        if dim < dim_range[0] or dim > dim_range[1]:
            ERROR(f"{self}'s unsqueeze sets dim={dim} out of rank={dim_range}")
        if dim < 0:
            dim = self.shape.dim() + dim + 1
        if not all([ss.is_constant for ss in self.shape.symbol]):
            shape = list(self.shape.symbol)
        else:
            shape = list(self.shape)
        shape = shape[:dim] + [1] + shape[dim:]
        return self.reshape(shape)

    def squeeze(self, dim: int):
        from AIPUBuilder import ops
        from AIPUBuilder.logger import ERROR
        dim_range = [-self.shape.dim() - 1, self.shape.dim() + 1]
        if dim < dim_range[0] or dim > dim_range[1]:
            ERROR(f"{self}'s squeeze sets dim={dim} out of rank={dim_range}")
        if dim < 0:
            dim = self.shape.dim() + dim + 1
        shape = list(self.shape)
        if shape[dim] != 1:
            return self
        shape = shape[:dim] + shape[dim+1:]
        return self.reshape(shape)

    def __getitem__(self, i):
        from AIPUBuilder import ops

        def slice_to_num(slice, max_len):
            begin = 0
            end = max_len
            step = 1
            if slice.start is not None:
                s = slice.start
                if s < 0:
                    s += max_len
                if s >= max_len:
                    raise IndexError("Index out of range.")
                begin = s
            if slice.stop is not None:
                s = slice.stop
                if s < 0:
                    s += max_len
                if s > max_len:
                    raise IndexError("Index out of range.")
                end = s
            if slice.step is not None:
                s = slice.step
                if s >= max_len or s <= -max_len:
                    raise IndexError("Index out of range.")
                step = s
            return begin, end, step
        full = slice(None)
        if isinstance(i, int):
            if i < 0:
                i = i + self.shape[0]
            if i >= self.shape[0]:
                raise IndexError("index out of range")
            if len(self.shape) != 1:
                i = tuple([i] + [full] * (len(self.shape) - 1))
            else:
                i = (i,)
        elif isinstance(i, slice):
            if len(self.shape) != 1:
                i = tuple([i] + [full] * (len(self.shape) - 1))
            else:
                i = (i,)

        if isinstance(i, tuple) and all([isinstance(k, (int, slice)) for k in i]) and len(i) == len(self.shape):
            begin = []
            end = []
            step = []
            for index, s in enumerate(i):
                if isinstance(s, int):
                    begin.append(s)
                    end.append(s + 1)
                    step.append(1)
                else:
                    be, en, st = slice_to_num(s, self.shape[1])
                    begin.append(be)
                    end.append(en)
                    step.append(st)
            return ops.strided_slice(self, begin, end, step)
        else:
            raise IndexError("Tensor does not support indexing with:", i)

    def eval(self):
        if self.op is None:
            return self.betensor
        for inp in self.op.inputs:
            inp.eval()
        from AIPUBuilder.Optimizer.framework import OP_DICT
        return OP_DICT[self.op.type](self.op)

    def to(self, device=0):
        # TODO
        from AIPUBuilder.driver import is_run_on_aipu, DevMemObj
        if not is_run_on_aipu():
            raise NotImplementedError("Tensor.to() is only supported on AIPU")
        dev = DevMemObj(self.mem_size())
        self._attrs['_dev'] = dev
        return self

    def set_data(self, v):
        import numpy as np
        self._set_data(np.ascontiguousarray(v).view(np.uint8))


@continue_class
class ComputeTensor:

    @property
    def np(self):
        return self.numpy()

    def numpy(self):
        if self.tensor is None:
            return None
        return self.tensor.numpy()
