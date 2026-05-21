#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3
from AIPUBuilder._C._qtlib import (QuantizationTransform,
                                   Quantizer,
                                   Dequantizer,
                                   to_fp24,
                                   InsertCastOp,
                                   deduce_float_strategy,
                                   simplify_qdq)
from AIPUBuilder._C._core import Scheme, Granularity, QuantizedRange, Quantization
from AIPUBuilder.core.utils import continue_class
from AIPUBuilder.logger import INFO, WARN, ERROR, DEBUG
from AIPUBuilder.core import *

__all__ = [
    "quantize_transform",
    "transform",
    "qt_quantize",
    "forward",
    "Quantizer",
    "Dequantizer",
    "Scheme",
    "Granularity",
    "QuantizedRange",
    "combine_quantization_method",
    "to_fp24",
    "cast_dtypes_for_lib",
    "simplify_qdq",
    "deduce_float_strategy",
]


def cast_dtypes_for_lib(g):
    inserter = InsertCastOp()
    inserter.insert(g)


@continue_class
class Granularity:
    def __str__(self):
        return self.__to_str__()


@continue_class
class Scheme:
    def __str__(self):
        return self.__to_str__()


@continue_class
class QuantizedRange:
    def __str__(self):
        return self.__to_str__()


@continue_class
class Quantization:

    @property
    def qscales(self):
        return self._scales()

    @qscales.setter
    def qscales(self, value):
        from AIPUBuilder.core import ComputeTensor
        if not isinstance(value, ComputeTensor):
            value = ComputeTensor(value)
        self._set_scales(value)

    @property
    def qoffsets(self):
        return self._offsets()

    @qoffsets.setter
    def qoffsets(self, value):
        from AIPUBuilder.core import ComputeTensor
        if not isinstance(value, ComputeTensor):
            value = ComputeTensor(value)
        self._set_offsets(value)

    @property
    def scales(self):
        import numpy as np
        r = self._scales().np
        if r is None:
            return np.array([])
        return r

    @scales.setter
    def scales(self, value):
        from AIPUBuilder.core import ComputeTensor
        self._set_scales(ComputeTensor(value))

    @property
    def offsets(self):
        import numpy as np
        r = self._offsets().np
        if r is None:
            return np.array([])
        return r

    @offsets.setter
    def offsets(self, value):
        from AIPUBuilder.core import ComputeTensor
        self._set_offsets(ComputeTensor(value))


def _build_qtransform(g, run_mode, quantization_bits, lut_items_in_bits):
    mode_dict = {
        'sym': 'Sym',
        'asym': 'Asym',
        'fasym': 'FAsym',
        'fsym': 'FSym',
        'ffasym': 'FFAsym',
        'pfasym': 'PFAsym',
        '8bsymweightunchange': 'FAsym',
        '8basymweightto16bsymweight': 'FSym'}

    if run_mode.lower() not in mode_dict.keys():
        ERROR(f"mode = {run_mode} is not supported. Now support: {mode_dict.keys()}")
    if not isinstance(lut_items_in_bits, int):
        try:
            lut_items_in_bits = int(lut_items_in_bits)
        except Exception as e:
            ERROR(f"{e}: lut_items_in_bits should be int, but now is {lut_items_in_bits}")
    mode = mode_dict[run_mode.lower()]
    # For legacy ops, ops will set aipu_dtype
    for n in g.nodes:
        for t in n.outputs:
            # t.quantization.primitive_dtype = t.dtype
            if mode.lower().endswith('fasym'):
                t.quantization.aipu_dtype = t.dtype
            else:
                t.quantization.aipu_dtype = Dtype.VOID
        for ck, ct in n.constants.items():
            # ct.quantization.primitive_dtype = ct.dtype
            if mode.lower().endswith('fasym') and ck in ["weights"]:
                if ct.quantization.offset == 0:
                    ct.quantization.aipu_dtype = ct.dtype
                else:
                    ct.quantization.aipu_dtype = Dtype.VOID
            else:
                ct.quantization.aipu_dtype = Dtype.VOID
            if mode.lower().endswith('fsym') and ck in ["weights"]:
                if ct.quantization.bits == -1:
                    ct.quantization.bits = 16
        if "weights" in n.constants and "biases" in n.constants:
            if n.constants["weights"].quantization.aipu_dtype != Dtype.VOID:
                # if n.constants["weights"].quantization.aipu_dtype != Dtype.VOID and n.constants['biases'].dtype.is_float:
                n.constants["biases"].quantization.aipu_dtype = n.constants["biases"].dtype

    for n in g.nodes:
        if 'lut_items_in_bits' not in n.attrs:
            n.attrs.update({"lut_items_in_bits": NodeParamValue(lut_items_in_bits)})

        for ot in n.outputs:
            if ot.quantization.lut_items_bits == -1:
                if isinstance(n.attrs["lut_items_in_bits"], NodeParamValue):
                    ot.quantization.lut_items_bits = n.attrs["lut_items_in_bits"].i
                else:
                    ot.quantization.lut_items_bits = n.attrs['lut_items_in_bits']
    qtransformer = QuantizationTransform(g, mode, quantization_bits)
    return qtransformer


def quantize_transform(g, run_mode='fASym', quantization_bits=8, lut_items_in_bits=8) -> bool:
    """
     Quantize the graph with specified run mode and lut bits.
      :param run_mode: default is 'Sym'. Choice bellow:
             . Sym
             . Asym
      :param lut_items_in_bits: default is 8. It also can be 8,10,12...
    """
    # # For legacy ops, ops will set aipu_dtype
    # for n in g.nodes:
    #     for t in n.outputs + list(n.constants.values()):
    #         if run_mode.lower().endswith('fasym') or run_mode.lower() in ['8bsymweightunchange', '8basymweighto16bsymweight']:
    #             t.quantization.aipu_dtype = t.dtype
    #         else:
    #             t.quantization.aipu_dtype = Dtype.VOID

    qtransformer = _build_qtransform(g, run_mode, quantization_bits, lut_items_in_bits)
    qtransformer.run()
    return True


def transform(g, run_mode='Sym', quantization_bits=8, lut_items_in_bits=8) -> bool:
    """
     transform the graph with specified run mode and lut bits from other quantization to AIPU quantization.
      :param run_mode: default is 'Sym'. Choice bellow:
             . Sym
             . Asym
      :param lut_items_in_bits: default is 8. It also can be 8,10,12...
    """
    qtransformer = _build_qtransform(g, run_mode, quantization_bits, lut_items_in_bits)
    qtransformer.transform()
    return True


def qt_quantize(g, run_mode='Sym', quantization_bits=8, lut_items_in_bits=8) -> bool:
    """
     Quantize the graph with specified run mode and lut bits from statistic min/max of each tensor.
      :param run_mode: default is 'Sym'. Choice bellow:
             . Sym
             . Asym
      :param lut_items_in_bits: default is 8. It also can be 8,10,12...
    """
    qtransformer = _build_qtransform(g, run_mode, quantization_bits, lut_items_in_bits)
    qtransformer.quantize()
    return True


def forward(g, run_mode='Sym', quantization_bits=8, lut_items_in_bits=8) -> bool:
    """
     Forward the graph with input deduecd from max and min in input tensor.
      :param run_mode: default is 'Sym'. Choice bellow:
             . Sym
             . Asym
      :param lut_items_in_bits: default is 8. It also can be 8,10,12...
    """
    qtransformer = _build_qtransform(g, run_mode, quantization_bits, lut_items_in_bits)
    qtransformer.forward()
    return True


def combine_quantization_method(qinfo):
    if not isinstance(qinfo, Quantization):
        return

    granu = qinfo.granularity.name.lower()
    range_mode = qinfo.quantized_range.name.lower()
    scheme = qinfo.scheme.name.lower()

    return f"{granu}_{scheme}_{range_mode}"  # per_tensor_symmetric_full_range
