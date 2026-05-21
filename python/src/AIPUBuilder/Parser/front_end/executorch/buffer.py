# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.
import inspect
import numpy as np
import re
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2, float8_e4m3fnuz, float8_e5m2fnuz, int4
from collections import OrderedDict
from ...common.utils import is_file, unpack_u8_to_4bit
from ...utils._optional_import import require_optional_import
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL

from .executorch_flatbuffer.Program import Program

# BYTE = 0,
# CHAR = 1,
# SHORT = 2,
# INT = 3,
# LONG = 4,
# HALF = 5,
# FLOAT = 6,
# DOUBLE = 7,
# BOOL = 11,
# QINT8 = 12,
# QUINT8 = 13,
# QINT32 = 14,
# BFLOAT16 = 15,
# QUINT4X2 = 16,
# QUINT2X4 = 17,
# BITS16 = 22,
# FLOAT8E5M2 = 23,
# FLOAT8E4M3FN = 24,
# FLOAT8E5M2FNUZ = 25,
# FLOAT8E4M3FNUZ = 26,
# UINT16 = 27,
# UINT32 = 28,
# UINT64 = 29,
# // Types currently not implemented.
# // COMPLEXHALF = 8,
# // COMPLEXFLOAT = 9,
# // COMPLEXDOUBLE = 10,
# // BITS1x8 = 18,
# // BITS2x4 = 19,
# // BITS4x2 = 20,
# // BITS8 = 21,

xnode_type_to_name = None

executorch_tensor_type_map = {
    0: np.uint8,
    1: np.int8,
    2: np.int16,
    3: np.int32,
    4: np.int64,
    5: np.float16,
    6: np.float32,
    7: np.float64,
    11: bool,
    15: bfloat16,
    23: float8_e5m2,
    24: float8_e4m3fn,
    25: float8_e5m2fnuz,
    26: float8_e4m3fnuz,
    27: np.uint16,
    28: np.uint32,
    29: np.uint64,
}

xnn_tensor_type_map = {
    1: np.float32,
    2: np.float16,
    3: np.int8,    # qint8, Quantized 8-bit signed integer with shared per-Value quantization parameters.
    4: np.uint8,   # quint8, Quantized 8-bit unsigned integer with shared per-Value quantization parameters.
    5: np.int32,   # qint32, Quantized 32-bit signed integer with shared per-Value quantization parameters.
    6: np.int8,   # qcint8, Quantized 8-bit signed integer with shared per-channel quantization parameters.
    7: np.int32,  # qcint32, Quantized 32-bit signed integer with shared per-channel quantization parameters.
    8: int4,   # qcint4, Quantized 4-bit signed integer with shared per-channel quantization parameters.
    9: np.int8,   # qdint8, Dynamically quantized 8-bit signed integer with per-batch quantization parameters.
    10: int4,  # qbint4, Quantized 4-bit signed integer with shared blockwise quantization parameters.
    # qpint8, Dynamically quantized 8-bit signed integers packed with their per-row quantization parameters.
    11: np.int8,
    12: np.int32,
    13: 'pfp32',   # IEEE754 single-precision packed floating-point.
    14: bfloat16,
}


def get_torch_op_schema(op_name, overload_name):
    # Aten ops
    torch = require_optional_import('torch', 'executorch schema parsing')
    schema = None
    schemas = torch._C._jit_get_schemas_for_operator(op_name)
    if schemas:
        for sch in schemas:
            if sch.name == op_name and sch.overload_name == overload_name:
                schema = sch
                break
    else:
        from .torch_metadata import gen_custom_torch_schema
        schema = gen_custom_torch_schema(op_name, overload_name)
    return schema


def parse_torch_type(type_obj):
    is_list, ele_type = False, type_obj

    def unwrap_optional(_type_obj):
        if _type_obj.kind() == 'OptionalType':
            inner_type = _type_obj.getElementType()
            return inner_type
        return _type_obj

    if hasattr(type_obj, 'kind'):
        unwrapped = unwrap_optional(type_obj)
        kind = unwrapped.kind()
        is_list = kind == 'ListType'
        ele_type = unwrapped.getElementType() if is_list else unwrapped

    return is_list, ele_type


class ExecuTorch(object):
    def __init__(self, buffer):
        self.name = 'ExecuTorch_model'
        self.buffer = buffer
        self.program = None
        self.ep_dict = {}
        self.fwd_ep = None
        self.fwd_inst_info = {}   # inst_id: {'inputs'/'outputs'}
        self.all_output_ids = set()

    def blob(self, offset, size):
        start = offset + self.segment_base_offset
        end = start + size
        data = self.buffer[start:end]
        return data

    def segment(self, key):
        named_data = self.get_named_data()
        if key in named_data:
            segment_idx = named_data[key]
            if 0 <= segment_idx < self.program.SegmentsLength():
                segment = self.program.Segments(segment_idx)
                offset = segment.Offset()
                size = segment.Size()
                data = self.blob(offset, size)
                return data
            else:
                return None
        else:
            return None

    def parse_header(self):
        # refer: https://docs.pytorch.org/executorch/main/pte-file-format.html
        assert len(self.buffer) >= 40, 'ExecuTorch buffer length incorrect!'
        assert self.buffer[4:8].decode() == 'ET12', 'ExecuTorch header magic incorrect!'
        # 4: Extended Header magic 4: Header length 8: Flatbuffer data size 8: Segment base offset 8: Segment data size
        expected_length = 4 + 4 + 8 + 8 + 8
        if len(self.buffer[8:]) < expected_length:
            ERROR('Not enough bytes for extended header!')
        magic = self.buffer[8:12]
        if magic.decode() == 'eh00':
            self.extended_header_size = int.from_bytes(self.buffer[12:16], byteorder='little')
            # size of the flatbuffer-encoded program data, starting from byte offset zero above.
            # I.e., it includes these headers.
            self.program_size = int.from_bytes(self.buffer[16:24], byteorder='little')
            self.segment_base_offset = int.from_bytes(self.buffer[24:32], byteorder='little')
        else:
            self.extended_header_size = 0
            self.program_size = 0
            self.segment_base_offset = 0

    def parse_program(self):
        self.program = Program.GetRootAs(self.buffer, 0)
        if self.program.ExecutionPlanLength() > 0:
            for i in range(self.program.ExecutionPlanLength()):
                ep = self.program.ExecutionPlan(i)
                ep_name = ep.Name().decode()
                self.ep_dict[ep_name] = ep
                if ep_name == 'forward':
                    self.fwd_ep = ep
            if self.fwd_ep.OperatorsLength() > 0 or self.fwd_ep.ChainsLength() > 0:
                pass
            else:
                WARN('[Parser]: Empty operators in executorch model!')
        else:
            WARN('[Parser]: Meets invalid executorch model in parse program!')

    def parse_value(self, value_idx):
        assert value_idx < self.fwd_ep.ValuesLength(), f'Incorrect value index in get_value_from_index!'
        evalue = self.fwd_ep.Values(value_idx)
        val_type = evalue.ValType()
        val = evalue.Val()
        value_info = {'name': str(value_idx), 'quantize': False, 'is_const': False}

        from .executorch_flatbuffer.KernelTypes import KernelTypes
        if val_type == KernelTypes.Tensor:
            from .executorch_flatbuffer.Tensor import Tensor
            parsed_value = Tensor()
            parsed_value.Init(val.Bytes, val.Pos)
            tensor_info = self.parse_tensor(parsed_value)
            value_info.update(tensor_info)
        elif val_type == KernelTypes.TensorList:
            from .executorch_flatbuffer.TensorList import TensorList
            t_list = TensorList()
            t_list.Init(val.Bytes, val.Pos)
            t_id_list = t_list.ItemsAsNumpy().tolist()
            value_info_list = []
            for t_id in t_id_list:
                _value_info = self.parse_value(t_id)
                value_info_list.append(_value_info)
            return value_info_list
        elif val_type == KernelTypes.OptionalTensorList:
            from .executorch_flatbuffer.OptionalTensorList import OptionalTensorList
            t_list = OptionalTensorList()
            t_list.Init(val.Bytes, val.Pos)
        elif val_type == KernelTypes.Null:
            value_info['data'] = None
            value_info['dtype'] = None
        elif val_type == KernelTypes.Int:
            from .executorch_flatbuffer.Int import Int
            parsed_value = Int()
            parsed_value.Init(val.Bytes, val.Pos)
            val = parsed_value.IntVal()
            value_info['data'] = np.array(val, dtype=np.int64)
            value_info['dtype'] = 'int64'
            value_info['shape'] = []
            value_info['is_const'] = True
        elif val_type == KernelTypes.Bool:
            from .executorch_flatbuffer.Bool import Bool
            parsed_value = Bool()
            parsed_value.Init(val.Bytes, val.Pos)
            val = parsed_value.BoolVal()
            value_info['data'] = np.array(val, dtype=bool)
            value_info['dtype'] = 'bool'
            value_info['shape'] = []
            value_info['is_const'] = True
        elif val_type == KernelTypes.Double:
            from .executorch_flatbuffer.Double import Double
            parsed_value = Double()
            parsed_value.Init(val.Bytes, val.Pos)
            val = parsed_value.DoubleVal()
            value_info['data'] = np.array(val, dtype=np.float64)
            value_info['dtype'] = 'float64'
            value_info['shape'] = []
            value_info['is_const'] = True
        elif val_type == KernelTypes.String:
            pass
        elif val_type == KernelTypes.IntList:
            from .executorch_flatbuffer.IntList import IntList
            parsed_value = IntList()
            parsed_value.Init(val.Bytes, val.Pos)
            t_id_list = parsed_value.ItemsAsNumpy().tolist()
            data_list = []
            for t_id in t_id_list:
                _value_info = self.parse_value(t_id)
                data_list.append(_value_info['data'])
            value_info['data'] = np.array(data_list, dtype=np.int64)
            value_info['dtype'] = 'int64'
            value_info['shape'] = [len(data_list)]
            value_info['is_const'] = True
        elif val_type == KernelTypes.DoubleList:
            from .executorch_flatbuffer.DoubleList import DoubleList
            parsed_value = DoubleList()
            parsed_value.Init(val.Bytes, val.Pos)
            t_id_list = parsed_value.ItemsAsNumpy().tolist()
            data_list = []
            for t_id in t_id_list:
                _value_info = self.parse_value(t_id)
                data_list.append(_value_info['data'])
            value_info['data'] = np.array(data_list, dtype=np.float64)
            value_info['dtype'] = 'float64'
            value_info['shape'] = [len(data_list)]
            value_info['is_const'] = True
        elif val_type == KernelTypes.BoolList:
            from .executorch_flatbuffer.BoolList import BoolList
            parsed_value = BoolList()
            parsed_value.Init(val.Bytes, val.Pos)
            t_id_list = parsed_value.ItemsAsNumpy().tolist()
            data_list = []
            for t_id in t_id_list:
                _value_info = self.parse_value(t_id)
                data_list.append(_value_info['data'])
            value_info['data'] = np.array(data_list, dtype=bool)
            value_info['dtype'] = 'bool'
            value_info['shape'] = [len(data_list)]
            value_info['is_const'] = True
        else:
            ERROR(f'Error Kernel Type: {val_type}!')
        return value_info

    def parse_chain(self):
        nodes = []
        for i in range(self.fwd_ep.ChainsLength()):
            chain = self.fwd_ep.Chains(i)
            for j in range(chain.InstructionsLength()):
                instruction = chain.Instructions(j)
                instrargs_type = instruction.InstrArgsType()
                instrargs = instruction.InstrArgs()
                instr_id = f'{i}_{j}'
                from .executorch_flatbuffer.InstructionArguments import InstructionArguments
                if instrargs_type == InstructionArguments.KernelCall:
                    from .executorch_flatbuffer.KernelCall import KernelCall
                    kernel = KernelCall()
                    kernel.Init(instrargs.Bytes, instrargs.Pos)
                    op_idx = kernel.OpIndex()
                    args = kernel.ArgsAsNumpy().tolist()
                    kernel_node = self.parse_kernel(instr_id, op_idx, args)
                    nodes.append(kernel_node)
                elif instrargs_type == InstructionArguments.DelegateCall:
                    from .executorch_flatbuffer.DelegateCall import DelegateCall
                    delegate_call = DelegateCall()
                    delegate_call.Init(instrargs.Bytes, instrargs.Pos)
                    delegate = self.fwd_ep.Delegates(delegate_call.DelegateIndex())
                    args = delegate_call.ArgsAsNumpy().tolist()
                    delegate_nodes = self.parse_delegate(instr_id, delegate, args)
                    nodes.extend(delegate_nodes)
                elif instrargs_type == InstructionArguments.MoveCall:
                    pass
                elif instrargs_type == InstructionArguments.JumpFalseCall:
                    pass
                else:
                    pass
        return nodes

    def parse_kernel(self, instr_id, op_idx, args):
        op = self.fwd_ep.Operators(op_idx)
        op_name = op.Name().decode()
        op_type = op_name[0].upper() + op_name[1:]
        op_type = op_type.replace('::', '')
        pure_op_name = op_name.split('::')[1]
        op_overload = op.Overload().decode()
        schema = get_torch_op_schema(op_name, op_overload)
        if schema is None:
            ERROR(f'Cannot find torch schema of op: {op_name}!.')
        node_attr = {
            'name': f'{pure_op_name}_{args[-1]}',
            'domain': instr_id,
            'type': op_type
        }
        args_info = {}
        for arg in args:
            if arg not in args_info:
                value_info = self.parse_value(arg)
                args_info[arg] = value_info
        inputs_len = len(schema.arguments)
        return_outputs = {list(arg.alias_info.before_set)[0] for arg in schema.returns
                          if arg.alias_info and len(arg.alias_info.before_set) == 1}
        input_ids = args[:inputs_len]
        output_ids = args[inputs_len:]
        new_input_ids = []
        for i in range(inputs_len):
            argument = schema.arguments[i]
            if argument.alias_info and len(argument.alias_info.before_set) == 1:
                if list(argument.alias_info.before_set)[0] in return_outputs:
                    continue
            # inp_name = argument.name
            is_list, ele_type = parse_torch_type(argument.type)
            if is_list:
                arg_info = args_info[input_ids[i]]
                torch = require_optional_import('torch', 'executorch operator parsing')
                if isinstance(ele_type, torch.TensorType):
                    for a_info in arg_info:
                        t_id = int(a_info['name'])
                        new_input_ids.append(t_id)
                        args_info[t_id] = a_info
                else:
                    new_input_ids.append(input_ids[i])
            else:
                new_input_ids.append(input_ids[i])
        inputs = []
        outputs = []
        for i in new_input_ids:
            input_info = args_info[i]
            if i in self.all_output_ids:
                input_info['is_const'] = False
            inputs.append(input_info)
        for i in output_ids:
            self.all_output_ids.add(i)
            out_info = args_info[i]
            out_info['is_const'] = False
            outputs.append(out_info)
        node_attr['inputs'] = inputs
        node_attr['outputs'] = outputs
        node_attr['op_attr'] = {}
        return node_attr

    def parse_delegate(self, instr_id, delegate, args):
        location = delegate.Processed().Location()
        from .executorch_flatbuffer.DataLocation import DataLocation
        if location == DataLocation.INLINE:
            data = self.program.BackendDelegateData(delegate.Processed().Index()).DataAsNumpy()
        else:
            # SEGMENT
            segment = self.program.Segments(delegate.Processed().Index())
            offset = segment.Offset()
            size = segment.Size()
            data = self.blob(offset, size)
        id = delegate.Id().decode('utf-8')
        if id == 'XnnpackBackend':
            xnnpack = XnnPack(instr_id, data)
            xnn_nodes = self.parse_xnnpack(xnnpack, args)
            self.fwd_inst_info[instr_id] = {'inputs': xnnpack.inputs, 'outputs': xnnpack.outputs}
            return xnn_nodes

    def parse_xnnpack(self, xnnpack, args):
        # check header
        xnnpack.parse_header()
        xnnpack.parse_graph()

        xnn_nodes = xnnpack.parse_nodes(self, args)
        return xnn_nodes

    def get_named_data(self):
        # key: segment_idx
        named_data = {}
        for i in range(self.program.NamedDataLength()):
            ndata = self.program.NamedData(i)
            named_data[ndata.Key().decode('utf-8')] = ndata.SegmentIndex()
        return named_data

    def parse_tensor(self, tensor):
        tensor_info = {}
        dtype = executorch_tensor_type_map[tensor.ScalarType()]
        shape = tensor.SizesAsNumpy().tolist()
        buf_idx = tensor.DataBufferIdx()
        alloc_info = tensor.AllocationInfo()
        extra_tensor_info = tensor.ExtraTensorInfo()
        if extra_tensor_info:
            raise NotImplementedError('Still not support parse extra_tensor_info')

        if buf_idx > 0:
            tensor_info['is_const'] = True
            data = np.frombuffer(self.parse_const(buf_idx, alloc_info), dtype=dtype).reshape(shape)
        else:
            tensor_info['is_const'] = False
            data = None
        tensor_info['data'] = data
        tensor_info['shape'] = shape
        tensor_info['dtype'] = str(dtype.__name__)
        tensor_info['dim_order'] = tensor.DimOrderAsNumpy().tolist()
        # 1.dynamism == STATIC: sizes field represents the static shape of the tensor.
        # 2.dynamism == DYNAMIC_BOUND: sizes field represents the upper bound shape of the tensor.Each
        # dimension of the tensor at runtime should never exceed the corresponding dimension of the
        # upper bound shape.
        # 3.dynamism == DYNAMIC_UNBOUND: the stored sizes field can be ignored since shape is fully dynamic.
        tensor_info['dynamic_type'] = tensor.ShapeDynamism()
        return tensor_info

    def parse_const(self, data_buf_index, alloc_info):
        const_data = None
        if not self.program.ConstantBufferIsNone() and self.program.ConstantBufferLength() > 0:
            assert data_buf_index < self.program.ConstantBufferLength(), 'Error const data buffer index out of range!'
            const_buffer = self.program.ConstantBuffer(data_buf_index)
            const_data = const_buffer.StorageDataAsNumpy()
        elif alloc_info is None:
            const_segment = self.program.ConstantSegment()
            data_segment = self.program.Segments(const_segment.SegmentIndex())
            offset = const_segment.Offsets(data_buf_index)
            next = data_segment.Size()
            if data_buf_index + 1 < const_segment.OffsetsLength():
                next = const_segment.Offsets(data_buf_index + 1)
            size = next - offset
            position = data_segment.Offset() + offset
            const_data = self.blob(position, size)
        else:
            raise NotImplementedError('Still not support tensor allocation info!')
        return const_data


class XnnPack(object):
    def __init__(self, name, buffer):
        self.name = name
        self.buffer = buffer
        self.inputs = []
        self.outputs = []

    def parse_header(self):
        assert len(self.buffer) >= 30, 'XNNPACK buffer length incorrect!'
        assert self.buffer[4:8].decode('utf-8') == 'XH00', 'XNNPACK header magic incorrect!'
        self.buf_offset = int.from_bytes(self.buffer[10:14], byteorder='little')
        self.buf_size = int.from_bytes(self.buffer[14:18], byteorder='little')
        self.const_offset = int.from_bytes(self.buffer[18:22], byteorder='little')
        self.const_size = int.from_bytes(self.buffer[22:26], byteorder='little')

    def parse_graph(self):
        graph_data = self.buffer[self.buf_offset: self.buf_offset + self.buf_size]
        # check Graph magic
        assert graph_data[4:8].decode('utf-8') == 'XN01', 'XNNPACK Graph magic incorrect!'
        from .fb_xnnpack.XNNGraph import XNNGraph
        self.graph = XNNGraph.GetRootAs(graph_data)

    def parse_nodes(self, ext_model, args):
        from .fb_xnnpack.XNodeUnion import XNodeUnion
        xnn_1x1_ops = [
            XNodeUnion.XNNSoftmax, XNodeUnion.XNNSigmoid, XNodeUnion.XNNClamp, XNodeUnion.XNNFloor,
            XNodeUnion.XNNConvert, XNodeUnion.XNNGlobalAvgPooling2d, XNodeUnion.XNNSquareRoot,
            XNodeUnion.XNNCeiling, XNodeUnion.XNNHardswish, XNodeUnion.XNNNegate, XNodeUnion.XNNSquare,
            XNodeUnion.XNNAbs, XNodeUnion.XNNReciprocalSquareRoot, XNodeUnion.XNNLog, XNodeUnion.XNNGelu,
            XNodeUnion.XNNTanh, XNodeUnion.XNNExp, XNodeUnion.XNNSin, XNodeUnion.XNNCopy, XNodeUnion.XNNCos
        ]
        xnn_2x1_ops = [
            XNodeUnion.XNNAdd, XNodeUnion.XNNDiv, XNodeUnion.XNNMinimum, XNodeUnion.XNNMultiply,
            XNodeUnion.XNNSubtract, XNodeUnion.XNNMaximum, XNodeUnion.XNNPReLU, XNodeUnion.XNNBatchMatrixMultiply,
        ]
        xnn_cat_ops = [
            XNodeUnion.XNNConcatenate2, XNodeUnion.XNNConcatenate3, XNodeUnion.XNNConcatenate4,
            XNodeUnion.XNNConcatenate5
        ]
        xnn_conv_ops = [
            XNodeUnion.XNNConv2d, XNodeUnion.XNNDepthwiseConv2d, XNodeUnion.XNNConvTranspose2d
        ]
        xnn_pooling_ops = [
            XNodeUnion.XNNAvgPooling2d, XNodeUnion.XNNMaxPooling2d
        ]
        xnn_nodes = []
        inputs_len = self.graph.InputIdsLength()
        outputs_len = self.graph.OutputIdsLength()
        graph_inputs_id = self.graph.InputIdsAsNumpy().tolist()
        graph_outputs_id = self.graph.OutputIdsAsNumpy().tolist()
        graph_io_map = {}
        inst_inputs = []
        inst_outputs = []
        for index in range(inputs_len):
            graph_io_map[f'{self.name}_{graph_inputs_id[index]}'] = str(args[index])
            inst_inputs.append(str(args[index]))
        for index in range(outputs_len):
            graph_io_map[f'{self.name}_{graph_outputs_id[index]}'] = str(args[inputs_len + index])
            inst_outputs.append(str(args[inputs_len + index]))
        self.inputs = inst_inputs
        self.outputs = inst_outputs
        for i in range(self.graph.XnodesLength()):
            node_attr = {}
            node = self.graph.Xnodes(i)
            node_type = node.XnodeUnionType()
            node_union = node.XnodeUnion()
            op_type = get_xnode_name_by_value(node_type)
            node_attr['type'] = op_type
            node_attr['domain'] = self.name
            # maybe same tensor in different input port
            inputs = []
            outputs = []
            if node_type in xnn_1x1_ops:
                from .fb_xnnpack._XNNNode1x1 import _XNNNode1x1
                n = _XNNNode1x1()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputId()]
            elif node_type in xnn_2x1_ops:
                from .fb_xnnpack._XNNNode2x1 import _XNNNode2x1
                n = _XNNNode2x1()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.Input1Id(), n.Input2Id()]
                output_ids = [n.OutputId()]
            elif node_type in xnn_cat_ops:
                from .fb_xnnpack._XNNCat import _XNNCat
                n = _XNNCat()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.Input1Id(), n.Input2Id(), n.Input3Id(), n.Input4Id(), n.Input5Id()]
                output_ids = [n.OutputId()]
            elif node_type in xnn_conv_ops:
                from .fb_xnnpack._XNNNodeConv import _XNNNodeConv
                n = _XNNNodeConv()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.Input1Id(), n.FilterId(), n.BiasId()]
                output_ids = [n.OutputId()]
            elif node_type in xnn_pooling_ops:
                from .fb_xnnpack._XNNPooling2D import _XNNPooling2D
                n = _XNNPooling2D()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputId()]
            elif node_type == XNodeUnion.XNNFullyConnected:
                from .fb_xnnpack.XNNFullyConnected import XNNFullyConnected
                n = XNNFullyConnected()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.Input1Id(), n.FilterId(), n.BiasId()]
                output_ids = [n.OutputId()]
            elif node_type == XNodeUnion.XNNStaticTranspose:
                from .fb_xnnpack.XNNStaticTranspose import XNNStaticTranspose
                n = XNNStaticTranspose()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputId()]
            elif node_type == XNodeUnion.XNNStaticResizeBilinear2D:
                from .fb_xnnpack.XNNStaticResizeBilinear2D import XNNStaticResizeBilinear2D
                n = XNNStaticResizeBilinear2D()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputId()]
            elif node_type == XNodeUnion.XNNStaticConstantPad:
                from .fb_xnnpack.XNNStaticConstantPad import XNNStaticConstantPad
                n = XNNStaticConstantPad()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputId()]
            elif node_type == XNodeUnion.XNNStaticReshape:
                from .fb_xnnpack.XNNStaticReshape import XNNStaticReshape
                n = XNNStaticReshape()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputId()]
            elif node_type == XNodeUnion.XNNArgMaxPooling2d:
                from .fb_xnnpack.XNNArgMaxPooling2d import XNNArgMaxPooling2d
                n = XNNArgMaxPooling2d()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputValueId(), n.OutputIndexId()]
            elif node_type == XNodeUnion.XNNLeakyReLU:
                from .fb_xnnpack.XNNLeakyReLU import XNNLeakyReLU
                n = XNNLeakyReLU()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputId()]
            elif node_type == XNodeUnion.XNNELU:
                from .fb_xnnpack.XNNELU import XNNELU
                n = XNNELU()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputId()]
            elif node_type == XNodeUnion.XNNStaticSlice:
                from .fb_xnnpack.XNNStaticSlice import XNNStaticSlice
                n = XNNStaticSlice()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.InputId()]
                output_ids = [n.OutputId()]
            elif node_type == XNodeUnion.XNNScaledDotProductAttention:
                from .fb_xnnpack.XNNScaledDotProductAttention import XNNScaledDotProductAttention
                n = XNNScaledDotProductAttention()
                n.Init(node_union.Bytes, node_union.Pos)
                op_attr = get_obj_attribute(n)
                input_ids = [n.QueryId(), n.KeyId(), n.ValueId(), n.ScaleId(), n.MaskId()]
                output_ids = [n.OutputId()]
            else:
                ERROR(f'Unexpected node type: {node_type}!')

            node_attr['name'] = f'{op_type}_{args[-1]}_{output_ids[-1]}'
            for _id in input_ids:
                tensor_info = self.parse_tensor(_id, ext_model)
                if tensor_info is not None:
                    inputs.append(tensor_info)
            for _id in output_ids:
                tensor_info = self.parse_tensor(_id, ext_model)
                if tensor_info is not None:
                    outputs.append(tensor_info)

            node_attr['inputs'] = inputs
            node_attr['outputs'] = outputs
            node_attr['op_attr'] = op_attr
            xnn_nodes.append(node_attr)

        # update backend deledate io tensor
        for node in xnn_nodes:
            for i, t in enumerate(node['inputs'][:]):
                if t['name'] in graph_io_map:
                    ext_id = int(graph_io_map[t['name']])
                    tensor_info = ext_model.parse_value(ext_id)
                    node['inputs'][i] = tensor_info
            for i, t in enumerate(node['outputs'][:]):
                if t['name'] in graph_io_map:
                    ext_id = int(graph_io_map[t['name']])
                    tensor_info = ext_model.parse_value(ext_id)
                    node['outputs'][i] = tensor_info
        return xnn_nodes

    def parse_tensor(self, tensor_id, ext_model):
        tensor_info = {
            'name': f'{self.name}_{tensor_id}',
            'is_const': False,
            'quantize': False
        }
        if tensor_id >= self.graph.XvaluesLength():
            return None
        xvalue = self.graph.Xvalues(tensor_id).XvalueUnion()
        xvalue_type = self.graph.Xvalues(tensor_id).XvalueUnionType()
        from .fb_xnnpack.XValueUnion import XValueUnion
        if xvalue_type == XValueUnion.XNNTensorValue:
            from .fb_xnnpack.XNNTensorValue import XNNTensorValue
            tensor_value = XNNTensorValue()
            tensor_value.Init(xvalue.Bytes, xvalue.Pos)
            xdata_type = xnn_tensor_type_map.get(tensor_value.Datatype(), None)
            shape = tensor_value.DimsAsNumpy().tolist() if tensor_value.DimsLength() > 0 else []
            const_buf_idx = tensor_value.ConstantBufferIdx()
            data = None
            if const_buf_idx > 0:
                tensor_info['is_const'] = True
                data = self.parse_const(const_buf_idx, ext_model)
            if data is None:
                parsed_data = None
            else:
                parsed_data = np.frombuffer(data, dtype=xdata_type).reshape(shape)
            tensor_info['data'] = parsed_data
            tensor_info['dtype'] = str(xdata_type.__name__)
            tensor_info['shape'] = shape
        elif xvalue_type == XValueUnion.XNNQuantizedTensorValue:
            from .fb_xnnpack.XNNQuantizedTensorValue import XNNQuantizedTensorValue
            from .fb_xnnpack.XNNQuantParams import XNNQuantParams
            quant_tensor_value = XNNQuantizedTensorValue()
            quant_tensor_value.Init(xvalue.Bytes, xvalue.Pos)
            tensor_value = quant_tensor_value.TensorValue()
            xdata_type = xnn_tensor_type_map.get(tensor_value.Datatype(), None)
            shape = tensor_value.DimsAsNumpy().tolist() if tensor_value.DimsLength() > 0 else []
            const_buf_idx = tensor_value.ConstantBufferIdx()
            data = None
            if const_buf_idx > 0:
                tensor_info['is_const'] = True
                data = self.parse_const(const_buf_idx, ext_model)
            if data is None:
                parsed_data = None
            else:
                if xdata_type is int4:
                    packed_data = np.frombuffer(data, dtype='uint8')
                    parsed_data = unpack_u8_to_4bit(packed_data, shape, 'int4')
                else:
                    parsed_data = np.frombuffer(data, dtype=xdata_type).reshape(shape)
            tensor_info['data'] = parsed_data
            tensor_info['dtype'] = str(xdata_type.__name__)
            tensor_info['shape'] = shape

            # handle quant params
            q_param_type = quant_tensor_value.QuantParamsType()
            q_param = quant_tensor_value.QuantParams()
            if q_param_type == XNNQuantParams.PerChannelQuant:
                from .fb_xnnpack.PerChannelQuant import PerChannelQuant
                quant = PerChannelQuant()
                quant.Init(q_param.Bytes, q_param.Pos)
                scale = quant.ScaleAsNumpy()
                channel_dim = quant.ChannelDim()
                tensor_info['quant_type'] = 'per_channel'
                tensor_info['scale'] = scale
                tensor_info['zp'] = 0
                tensor_info['channel_dim'] = channel_dim
            elif q_param_type == XNNQuantParams.PerTensorQuant:
                from .fb_xnnpack.PerTensorQuant import PerTensorQuant
                quant = PerTensorQuant()
                quant.Init(q_param.Bytes, q_param.Pos)
                scale = quant.Scale()
                zp = quant.ZeroPoint()
                tensor_info['quant_type'] = 'per_tensor'
                tensor_info['scale'] = scale
                tensor_info['zp'] = zp
            elif q_param_type == XNNQuantParams.PerTokenDynamicQuant:
                from .fb_xnnpack.PerTokenDynamicQuant import PerTokenDynamicQuant
                quant = PerTokenDynamicQuant()
                quant.Init(q_param.Bytes, q_param.Pos)
                tensor_info['quant_type'] = 'per_token_dynamic'
                tensor_info['non_batch_dim_nums'] = quant.NumNonbatchDims()
            else:
                from .fb_xnnpack.PerChannelGroupQuant import PerChannelGroupQuant
                quant = PerChannelGroupQuant()
                quant.Init(q_param.Bytes, q_param.Pos)
                scale = quant.ScaleAsNumpy()
                tensor_info['quant_type'] = 'per_group'
                tensor_info['scale'] = scale
                tensor_info['zp'] = 0
                tensor_info['channel_dim'] = quant.ChannelDim()
                tensor_info['group_size'] = quant.GroupSize()

            tensor_info['quantize'] = True
        else:
            ERROR(f'Unexpected xvalue type: {xvalue_type}!')
        return tensor_info

    def parse_const(self, const_id, ext_model):
        const_data = self.graph.ConstantData(const_id)
        named_key = const_data.NamedKey()
        if named_key:
            key_str = named_key.decode()
            data = ext_model.segment(key_str)
            return data
        else:
            offset = const_data.Offset()
            size = const_data.Size()
            offset = self.const_offset + offset
            data = self.buffer[offset: offset + size]
            return data


def get_xnode_name_by_value(xnode_type):
    from .fb_xnnpack.XNodeUnion import XNodeUnion
    global xnode_type_to_name
    if xnode_type_to_name is None:
        xnode_type_to_name = {}
        for attr_name, attr_value in XNodeUnion.__dict__.items():
            if not attr_name.startswith("__") and isinstance(attr_value, int):
                xnode_type_to_name[attr_value] = attr_name
    return xnode_type_to_name.get(xnode_type, "Unknown")


def read_executorch_model(pte_path):
    ret = None
    if is_file(pte_path):
        try:
            model_buf = open(pte_path, 'rb').read()
            ext_model = ExecuTorch(model_buf)
            ext_model.parse_header()
            ext_model.parse_program()

            ret = ext_model
        except IOError:
            ERROR('[Parser]: Meets error when reading executorch file(%s) in read_executorch_model!' %
                  pte_path)
    else:
        WARN('[Parser]: Invalid executorch file (%s) in read_executorch_model!' %
             pte_path)
    return ret


def get_obj_attribute(op):
    ret = {}
    attr_set = {v for v in dir(op) if not v.startswith(
        '_') and not v.startswith('GetRootAs') and v != 'Init'}
    for attr in attr_set:
        attr_call = getattr(op, attr)
        if callable(attr_call) and len(inspect.getfullargspec(attr_call)[0]) == 1:
            value = attr_call()
            ret.update({attr: value})
    return ret
