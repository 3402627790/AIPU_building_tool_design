import os
import re
import torch
from AIPUBuilder.core import aipugb, AIPUELF, LlmInfo, BuilderConfig
from AIPUBuilder.logger import INFO, ERROR, DEBUG, WARN
from AIPUBuilder.model.utils import string_to_base_type
from AIPUBuilder.model.loadweight import LoadWeight


class ModelBuild(object):
    def __init__(self, model, model_type, target, config):
        self.aipu_g = model
        self.model_type = model_type
        self.target = target
        self.config = config
        self.output_bin = None
        self.max_wt_size = config.get('max_wt_size', -1)
        self.compiled_g = None
        self.embedding_data = None

    def get_compiled_graph(self):
        return self.compiled_g

    def build(self, output_bin, enable_ds=True):
        self.output_bin = output_bin

        args_map = {
            'target': self.target,
            # 'op_plugin_options': 'AIPU_32BIT_BIAS=1,FORCE_BATCHFC_TO_CONV=1,FORCE_ADD_SUB_MUL_TO_ELTWISE=1',
            # 'op_plugin_options': 'AIPU_32BIT_BIAS=1,FORCE_ADD_SUB_MUL_TO_ELTWISE=1',
            # 'op_plugin_options': 'AIPU_32BIT_BIAS=1',
            # 'max_wt_size': 2048,
            'output': output_bin,

            # 'dumpIR': f"{self.config.model_name}_dumpir",
            # 'show_location': True,
        }

        if self.max_wt_size > -1:
            INFO(f"set max_wt_size = {self.max_wt_size}.")
            args_map.update({'max_wt_size': self.max_wt_size})

        if self.config.get('app', '') != '' and 'spm' in self.config.app[:self.config.app.index('(')].lower():
            args_map.update({'profile': True})

        BuilderConfig.reset()

        if self.config.get('builder_config', '') != '':
            bc = self.config.builder_config
            if (bc.startswith('"') and bc.endswith('"')) or (bc.startswith("'") and bc.endswith("'")):
                bc = bc[1:-1]
            pattern = r"--([a-zA-Z_][a-zA-Z0-9_-]*)=('(?:[^']|\\')*'|" + r'"(?:[^"]|\\")*"|[^,\s]+)'
            matches = re.findall(pattern, bc)
            bc_map = {}
            for match in matches:
                k = match[0]
                v = match[1]
                if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                    v = v[1:-1]
                v = string_to_base_type(v)
                if k not in bc_map:
                    bc_map[k] = [v]
                else:
                    bc_map[k].append(v)
            for bck, bcv in bc_map.items():
                if len(bcv) == 1:
                    if bck in args_map:
                        WARN(f"{bck} is already in args_map, the value will be overwritten")
                    args_map[bck] = bcv[0]
                elif len(bcv) > 1:
                    pre = ''
                    if bck in args_map:
                        WARN(f"{bck} is already in args_map, the value will be appended")
                        pre = f"{args_map.get(bck, '')},"
                        for one_bcv in bcv:
                            if one_bcv in pre:
                                bcv.remove(one_bcv)
                    args_map[bck] = pre + ','.join(bcv)
                else:
                    WARN(f"{bck} is not a valid key")

        if ('enable_ds' in args_map and args_map['enable_ds']) or 'enable_ds' not in args_map:
            args_map.update({'enable_ds': True})
            # args_map.update({'enable-pass': "split_to_crops"})
            # args_map.update({'disable-plugins': "HwaEltwiseDS"})
        elif 'enable_ds' in args_map:
            args_map.pop('enable_ds')

        INFO(f"\t Gbuilder compiles the IR with args:[{args_map}]")
        self.compiled_g = aipugb(self.aipu_g, args_map)

        ret = True if self.compiled_g else False
        return ret

    def __call__(self, output_bin, enable_ds):
        ret = self.build(output_bin, enable_ds)
        return ret
