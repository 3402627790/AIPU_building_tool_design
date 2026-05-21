from AIPUBuilder.logger import INFO
from AIPUBuilder import nn
from AIPUBuilder import ops
from AIPUBuilder.core import *  # noqa
import torch
import numpy as np

def conv2d_to_fc(g):
    def _conv_to_fc(node):
        def criteria(onen):
            if onen.type != OpType.Convolution:
                return False
            inp_shape = onen.inputs[0].shape
            out_shape = onen.outputs[0].shape
            max_kernel_size = 7
            kx = onen.params['kernel_x']
            ky = onen.params['kernel_y']
            sx = onen.params['stride_x']
            sy = onen.params['stride_y']
            dx = onen.params['dilation_x']
            dy = onen.params['dilation_y']
            grp = onen.params['group']
            pl = onen.params['pad_left']
            pr = onen.params['pad_right']
            pt = onen.params['pad_top']
            pb = onen.params['pad_bottom']
            ret = kx == sx and ky == sy
            ret = (kx > max_kernel_size or ky > max_kernel_size) and ret
            ret = (inp_shape.w == out_shape.w * kx) and ret
            ret = (inp_shape.h == out_shape.h * ky) and ret
            ret = (dx == 1 and dy == 1) and ret
            ret = grp == 1 and ret
            ret = (pl == 0 and pr == 0 and pt == 0 and pb == 0) and ret
            return ret

        if not criteria(node):
            return False

        INFO("tranform the conv with big kernel(kx or ky > 7) to batch fullyconnected in llm.")
        inp = node.inputs[0]
        out = node.outputs[0]
        kx = node.params['kernel_x']
        ky = node.params['kernel_y']
        sx = node.params['stride_x']
        sy = node.params['stride_y']
        dx = node.params['dilation_x']
        dy = node.params['dilation_y']
        grp = node.params['group']
        pl = node.params['pad_left']
        pr = node.params['pad_right']
        pt = node.params['pad_top']
        pb = node.params['pad_bottom']

        g = node.graph
        with g:
            bs_symbol, inh_symbol, inw_symbol, inc_symbol = inp.shape.symbol
            _, outh_symbol, outw_symbol, outc_symbol = out.shape.symbol
            new_in_shape = [bs_symbol, outh_symbol, ky, outw_symbol, kx, inc_symbol]
            reshape_o_t = inp.reshape(new_in_shape)
            trans_o_t = ops.transpose(reshape_o_t, [0, 1, 3, 2, 4, 5])
            fc_in_shape = [bs_symbol * outh_symbol * outw_symbol, ky * kx * inc_symbol]
            fc_in_t = trans_o_t.reshape(fc_in_shape)

            fc_bs, fc_inc = fc_in_t.shape
            fc = nn.Linear(fc_inc, out.shape[-1], dtype=out.dtype, name=node.name)
            fc_o_t = fc(fc_in_t)
            w = node.constants['weights'].betensor
            w = w.reshape(fc_o_t.shape[1], fc_in_t.shape[1])
            fc.weights.betensor = w
            fc.weights.sync_betensor()
            fc.weights.quantization = node.constants['weights'].quantization
            fc.biases.betensor = node.constants['biases'].betensor
            fc.biases.sync_betensor()
            fc.biases.quantization = node.constants['biases'].quantization

            reshape_fc_o_t = fc_o_t.reshape([bs_symbol, outh_symbol, outw_symbol, outc_symbol])

            consumers = out.consumers
            for con in consumers:
                idx = con.remove_input(out)
                con.add_input(reshape_fc_o_t, idx)
            node.clear_inputs()
            g.remove_node(node)
        return True

    def ctf(node):
        ret = False
        if SubGraphNode.is_subgraph_node(node):
            sbg = SubGraphNode.from_node(node).subgraph
            sbg_nodes = sbg.nodes[:]
            for sbn in sbg_nodes:
                ret = ctf(sbn) or ret
        else:
            ret = _conv_to_fc(node) or ret
        return ret

    nodes = g.nodes[:]
    ret = False
    for n in nodes:
        ret = ctf(n) or ret

    if ret:
        g.topological_sort()

def convert_group_fp16_scales_to_fp8(g, afp8_type):
    def _convert_fp16_scales_to_fp8_scales(unpacked_weight, scales, unpacked_zp, group_size, afp8_type='float8_e4m3fn'):
        from AIPUBuilder.Optimizer.utils.quant_tool_utils import linear_dequantize, cosine_distance
        from torch.nn import MSELoss as mseloss
        unpacked_weight = torch.tensor(unpacked_weight.astype(np.float32))
        scales = scales.reshape(unpacked_weight.shape[0], -1)
        scales = scales.repeat_interleave(group_size, dim=-1)
        qw_scale = 1.0/scales
        zeros = unpacked_zp.reshape(unpacked_weight.shape[0], -1)
        zeros = zeros.repeat_interleave(group_size, dim=-1)

        # for w8a8
        MSE = mseloss()
        # dequant weight
        fwt = (unpacked_weight.float() - zeros.float()) * qw_scale.float()
        fw_betensor_f16 = (fwt.to(torch.float32))
        # qU, qS_f16, qVh = torch.linalg.svd(fw_betensor_f16.float(), full_matrices=False)
        qS_f16 = torch.abs(torch.fft.fft(
            fw_betensor_f16.float(), norm='forward'))

        all_min_scale, all_org_max_scale = qw_scale.min(
            -1)[0], qw_scale.max(-1)[0]
        org_qw_scale = qw_scale
        min_dist = 1000
        if afp8_type == 'float8_e4m3fn':
            dtype = (torch.float8_e4m3fn)
            fmax = 448
        else:
            dtype = (torch.float8_e5m2)
            fmax = 57344
        channelnum = qw_scale.shape[0]
        sec_scale = torch.zeros(channelnum, 1)
        fp8_scales = torch.zeros_like(scales).to(dtype)
        for c in range(channelnum):
            min_dist = 1000
            # best_w = fw_betensor_f16[c]
            min_scale = all_min_scale[c]
            if 1.0/min_scale > fmax:
                beta = 1.525
                alpha = 0.625
                org_max_scale = all_org_max_scale[c]
            else:
                beta = 1.125
                alpha = 0.875
                org_max_scale = torch.tensor(1.0)
            for x in torch.linspace(alpha, beta, 10):
                # alpha = 0.8752
                max_scale = x*org_max_scale

                qw_scale = (org_qw_scale[c]/max_scale)
                c_min_scale = qw_scale.min()
                if 1.0/c_min_scale > fmax:
                    break
                dscale = max_scale.to(torch.float16)
                qw_scale_fp8 = (qw_scale).to(dtype)
                fwt = (unpacked_weight[c].float() - 0) * qw_scale_fp8.float()
                fwt_fp8 = (fwt).to(dtype)
                fw_betensor = (fwt_fp8.to(torch.float16)
                               * dscale).to(torch.float16)

                qS_f8 = torch.abs(torch.fft.fft(
                    fw_betensor.float(), norm='forward'))
                cos_sim = cosine_distance(qS_f8, qS_f16[c])
                dist = MSE(fw_betensor.float(), fw_betensor_f16[c])

                if dist < min_dist*cos_sim:
                    min_dist = dist
                    fp8_scales[c, ...] = qw_scale_fp8
                    sec_scale[c, ...] = dscale.float()

        return fp8_scales[:, ::group_size], sec_scale.to(torch.float16)

    def _convsert_fc_node_qinfo(n, afp8_type):
        def _delete_key_in_dict(dict,key):
            if key in dict:
                del dict[key]
            return
        # need re-quant activation,so need delete some quant params
        _delete_key_in_dict(n.params,'unquantifiable')
        if n.type in [OpType.Convolution, OpType.FullyConnected]:
            w = n.constants['weights']
            w.quantization.maxs =[]
            w.quantization.mins =[]
            if w.dtype== Dtype('float4_e2m1fn'):
                wshape = w.shape
                scales = torch.tensor(w.quantization.scales)
                scales_shape =scales.shape[0]
                group_size = (wshape[0]*wshape[1])//scales_shape
                zeros = torch.tensor(w.quantization.offsets.astype(np.float32))
                scale0,scale1 = _convert_fp16_scales_to_fp8_scales((w.np),scales,zeros,group_size,afp8_type)
                n.constants['scale0'] = Tensor('scale0',TensorShape([scale0.shape[0],scale0.shape[1]]),Dtype(afp8_type))
                n.constants['scale0'].dtype = Dtype(afp8_type)
                n.constants['scale0'].betensor = scale0
                n.constants['zp0'] = Tensor('zp0',TensorShape([scale0.shape[0],scale0.shape[1]]),Dtype('float4_e2m1fn'))
                n.constants['zp0'].betensor = zeros.reshape([scale0.shape[0],scale0.shape[1]])
                n.constants['scale1'] = Tensor(scale1.cpu().numpy().astype(np.float16))
                n.constants['scale1'].dtype = Dtype('float16')

        elif n.type == OpType.Softmax:
            _delete_key_in_dict(n.constants,'lut')
            _delete_key_in_dict(n.params,'lut_mode')
            _delete_key_in_dict(n.params,'is_perf_mode')
        elif n.type == OpType.Activation:
            _delete_key_in_dict(n.constants,'lut')
            _delete_key_in_dict(n.params,'lut_mode')
            _delete_key_in_dict(n.params,'is_perf_mode')
            _delete_key_in_dict(n.params,'index_offset_type')
            _delete_key_in_dict(n.params,'index_offset_value')
            _delete_key_in_dict(n.params,'index_scale_type')
            _delete_key_in_dict(n.params,'index_scale_value')
            _delete_key_in_dict(n.params,'value_offset_type')
            _delete_key_in_dict(n.params,'value_offset_value')
        return True

    def ctf(node):
        ret = False
        if SubGraphNode.is_subgraph_node(node):
            sbg = SubGraphNode.from_node(node).subgraph
            sbg_nodes = sbg.nodes[:]
            for sbn in sbg_nodes:
                ret = ctf(sbn) or ret
        else:
            ret = _convsert_fc_node_qinfo(node, afp8_type) or ret
        return ret

    nodes = g.nodes[:]
    ret = False
    for n in nodes:
        ret = ctf(n) or ret
