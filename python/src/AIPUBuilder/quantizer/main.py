# cython: language_level=3
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import configparser
import sys
from AIPUBuilder.core import Graph, Quantizer, quantize_transform
from AIPUBuilder.logger import INFO


def local_quantizer(g: Graph, scheme, qbits):
    q_eniger = Quantizer(g, scheme, qbits)
    q_eniger.quantize()
    return g


def quantize_transform_for_qat(g, run_mode):
    quantize_transform(g, run_mode)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-v', '--version', action='version',
                      version='0.1')
    args.add_argument("graph", metavar='<graph.def>',
                      type=str, help='graph definition file.')
    args.add_argument('-w', '--weight', type=str, default="",
                      help='the weight file for the graph.')
    args.add_argument('-o', '--output', type=str, default="output.def",
                      help='the optimized graph definition file.')
    args.add_argument('-b', '--bin', type=str, default="output.bin",
                      help='the optimized graph weight file.')

    args.add_argument('--bits', type=int, default=8,
                      help='the quantization bits of graph')
    args.add_argument('-s', '--scheme', type=str, default='symmetric',
                      help='the quantization scheme of graph: asymmetric or symmetric')
    options = args.parse_args(sys.argv[1:])

    g = Graph.parse(options.graph, options.weight)
    is_qat_model = 'compat_quantized_model' in g.attrs and g.attrs['compat_quantized_model']
    INFO(f"Start to {'quantization_transform for qat' if is_qat_model else 'quantize'} the graph...")
    if is_qat_model:
        quantize_transform_for_qat(g, '8bSymWeightUnchange')
        # quant_g = g
    else:
        quant_g = local_quantizer(g, options.scheme, options.bits)
        quant_g.serialize_scale_zp = True
        quant_g.serialize(options.output, options.bin)
    INFO("Quantize Done.")


if __name__ == "__main__":
    main()
