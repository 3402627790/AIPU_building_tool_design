import os
import re
import copy
import inspect
from AIPUBuilder.core import Graph, BuilderConfig
from AIPUBuilder.model.config import PretrainedConfig
from AIPUBuilder.logger import INFO, WARN, DEBUG, ERROR
from AIPUBuilder.model.aipullm import AIPULLM
from AIPUBuilder.model.qinfo import QuantType
from AIPUBuilder.model.loadweight import LoadWeight
from AIPUBuilder.model.utils import parse_string_to_pairs
from AIPUBuilder.model.register import APP, UPDATECONFIG, CAUSALLMS


class BuildPipeLine(object):

    def __init__(self, configs):
        self.configs = configs
        self.weight_cache = {}
        self.aipu_configs = []

    def __call__(self):
        for one_config_name, one_config in self.configs.items():
            aipu_config = self.gen_aipu_config(one_config, one_config_name if one_config_name != "Common" else "")
            self.build_one(aipu_config, one_config_name)
            self.aipu_configs.append(aipu_config)
        return self.aipu_configs

    def gen_aipu_config(self, config, config_name=''):
        '''
        update the configuration parameters to hf_config
        '''
        model_path = config.model_path
        model_type = config.model_name if config.model_name != "" else None

        if not os.path.exists(model_path):
            bp = os.environ.get("AIPU_LLM_MODEL_PATH", '/project/ai/scratch01/Model_Resource/huggingface')
            model_path = os.path.join(bp, model_path)
        if not os.path.exists(model_path):
            ERROR(f"the LLM model path is not exists, please check the path = {model_path}")

        if model_type is None:
            config_json_path = os.path.join(model_path, 'config.json')
            with open(config_json_path, 'r') as fi:
                lines = fi.readlines()
                for l in lines:
                    if 'model_type' in l:
                        model_type = l.split(':')[-1].strip().replace(' ', '')[1:-2]
        if model_type is None:
            ERROR("model type should be in config.json or setted by customer in cfg, but now is None")
            return

        INFO(f"model({model_type})'s model path = {model_path}")

        kwargs = config.__dict__
        _kvs = {
            'model_path': model_path,
            'model_type': model_type,
        }
        kwargs.update(_kvs)

        if os.path.exists(os.path.join(model_path, 'model.safetensors.index.json')):
            kwargs.update({'safetensor_index_json': os.path.join(model_path, 'model.safetensors.index.json')})

        if hasattr(config, 'vocab_path') and config.vocab_path != "":
            INFO(f"{config.model_name} has setted the vocab path in AIPULLM, the vocab json path is {config.vocab_path}")
            kwargs.update({'vocab_json': config.vocab_path})
        else:
            inherted_class_path = inspect.getfile(CAUSALLMS[model_type])
            model_type_vocab_json = os.path.join(os.path.dirname(
                inherted_class_path), f'{model_type}_vocab.json')
            if not os.path.exists(model_type_vocab_json):
                vocab_json = os.path.join(os.path.dirname(inherted_class_path), f'vocab.json')
                if not os.path.exists(vocab_json):
                    WARN(
                        f"vocab json path does not exists, please provide this file {config.model_type}_vocab.json, or vocab.json.")
                    model_type_vocab_json = ""
                else:
                    model_type_vocab_json = vocab_json
            kwargs.update({'vocab_json': model_type_vocab_json})

        if hasattr(config, 'quantization_config') and config.quantization_config != '':
            qc = config.quantization_config
            qc = qc.replace(' ', '').replace('"', '').replace("'", '')
            aipu_quant_config = parse_string_to_pairs(qc)
            if len(aipu_quant_config):
                kwargs.update({'aipu_quant_config': aipu_quant_config})

        quant_type = config.quant_type
        if quant_type != '':
            weight_activation_pair = QuantType.qmode_to_weight_and_activation_pair(quant_type)
            kwargs.update(**weight_activation_pair)
            # if QuantType.is_weight_only_quantization(quant_type):
            #     kwargs['quant_type'] = quant_type.lower().split('a')[0]

        weights = self.weight_cache.get(model_path, None)
        if weights is not None:
            kwargs.update({'weights': weights})

        if os.path.isfile(model_path):
            config_json_path = os.path.join(os.path.dirname(model_path), config.config_name)
        else:
            config_json_path = os.path.join(model_path, config.config_name)

        if config_name != '':
            kwargs.update({'config_name': config_name})

        config_json_path = '' if not os.path.exists(config_json_path) else config_json_path
        aipu_config = PretrainedConfig.from_json_file(config_json_path)
        aipu_config.update(kwargs)

        if aipu_config.hf_quantization_config is not None and aipu_config.hf_quantization_config.get('mode', None) is not None:
            aipu_config.hf_quantization_config.update({'quant_method': aipu_config.hf_quantization_config.get('mode')})
        # instance LoadWeight, and get llminfo. if safetensors and torchbin, will not case. gguf should update
        # the llminfo to config.
        loadweight_handler = LoadWeight(aipu_config)
        llminfo = loadweight_handler.llminfo()
        aipu_config.update({'weights': loadweight_handler})
        aipu_config.update(llminfo)
        if 'gguf_file_type' in aipu_config:
            aipu_config.hf_quantization_config = {'quant_method': 'gguf_quant'}

        if hasattr(config, 'option_config') and config.option_config != '':
            ocs = config.option_config.replace(' ', '').replace('"', '').replace("'", '')
            option_config_dict = parse_string_to_pairs(ocs)
            for k, v in aipu_config.items():
                # set multi-model's sub-section, like the parameters in vision_config
                if '_config' in k and isinstance(v, dict) and k != 'quantization_config':
                    v.update(option_config_dict)
            aipu_config.update(option_config_dict)

        if aipu_config.get('fill_llminfo', False) or 'cllmchat' in aipu_config.get('app', ''):
            aipu_config.update({'fill_llminfo': True})

        # model has difference parameters which need to be updated, because llminfo we use fixed parameter name.
        # like num_hidden_layers, in chatglm3, its num_layers.
        for reg_model_name, update_config_func in UPDATECONFIG.items():
            if reg_model_name in model_type or model_type in reg_model_name:
                inherted_class_path = inspect.getfile(CAUSALLMS[model_type])
                config_func_path = inspect.getfile(update_config_func)
                if inherted_class_path != config_func_path:
                    continue
                update_config_func(aipu_config)

        # if output_dir is not exist, will make dir first
        if aipu_config.get('output_dir', '') != '':
            output_dir = aipu_config.get('output_dir')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        aipu_binary_path = aipu_config.get('aipu_binary_path', '')
        if aipu_binary_path != '' and aipu_binary_path.endswith('.txt'):
            aipu_config.update({'quantized_ir_path': aipu_binary_path})

        return aipu_config

    def build_one(self, aipu_config, one_config_name=""):
        only_compile, skip_compile = False, False
        if aipu_config.get('aipu_binary_path', '') != "":
            if aipu_config.has_aipu_binary or (aipu_config.has_ir and aipu_config.engine in ["opt"]):
                INFO(
                    f"In {one_config_name} has configured the aipu binary path(={aipu_config.aipu_binary_path}), so skip the build-pipeline.")
                return
            only_compile = True
        if (aipu_config.get('engine', '')).lower() == "opt":
            skip_compile = True

        aipu_llm = AIPULLM.from_config(aipu_config)

        quantization_type = aipu_config.quant_type if aipu_config.quant_type != "" else None
        target = aipu_config.target
        if not only_compile:
            # BuilderConfig.set('enable_ds', False)
            _ = aipu_llm.load_huggingface()

            # cache already has readed safetensor
            model_path = aipu_config.model_path
            if model_path not in self.weight_cache and aipu_config.get("weights", False):
                self.weight_cache.update({model_path: aipu_config.weights})

            if aipu_config.only_fp_ir:
                INFO(f"{aipu_config.model_type} generates float IR Done.")
                return True
        else:
            irt = aipu_config.get('quantized_ir_path', '')
            irb = irt.replace('.txt', '.bin')
            aipu_llm.set_model(Graph.parse(irt, irb))
            quantization_type = None

        if aipu_config.only_qt_ir or target == "" or skip_compile:
            target = None

        ret = aipu_llm.build(quantization_type=quantization_type, target=target)
        if not ret:
            ERROR("[LLM] aipullm build process failed.")
            return False

        return True


class AppPipeLine(object):
    def __init__(self, aipu_configs, config_name=""):
        self.configs = aipu_configs
        self.config_name = config_name
        self.apps = []

    def _get_app_from_config(self):
        apps_dict = {}
        for aipu_config in self.configs:
            app_config = aipu_config.app
            if app_config not in apps_dict:
                apps_dict.update({app_config: []})
            apps_dict[app_config].append(aipu_config)
        return apps_dict

    def _parse(self, plugin):
        func_args = []
        if '(' not in plugin:
            func_args.append([plugin, {}])
        else:
            plugin_name = plugin[:plugin.index('(')].replace(' ', '').lower()
            if plugin.replace(' ', '') == f"{plugin_name}()":
                func_args.append([plugin_name, {}])
            else:
                right_parenthesis_pos = plugin.rindex(')')
                plugin_args = plugin[plugin.index('(') + 1: right_parenthesis_pos]
                arg_pattern = r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"'

                arg_matches = re.findall(arg_pattern, plugin_args)
                if arg_matches:
                    plugin_args = {}
                    for key, value in arg_matches:
                        plugin_args.update({key: value})
                        # print(f"  Key: {key}, Value: {value}")
                    func_args.append([plugin_name, plugin_args])

        INFO(f"[LLM] parse app configuration: {func_args}")
        return func_args

    def forward(self):
        app_plugins = self._get_app_from_config()
        for plugin, aipu_configs in app_plugins.items():
            if plugin == '':
                INFO("app=''(default value) will skip apppipeline.")
                continue
            plugin_name_and_args = self._parse(plugin)
            if len(plugin_name_and_args) != 1:
                ERROR(f"[LLM] found more than 1 app configuration in one app(={plugin})'")
            plugin_name, plugin_args = plugin_name_and_args[0]
            app = APP[plugin_name](**plugin_args)
            app.set_configs(aipu_configs)
            app.forward()
            self.apps.append(app)


class PipeLine(object):

    def __init__(self, config_maps):
        self.config_maps = config_maps

    def _config_preprocess(self, configs):
        model_types = []
        for _, config in configs.items():
            model_types.append(config.model_name)

            # set defualt value for ir_prefix
            config.__setattr__('ir_prefix', '')

            if config.app != "" and config.engine == "" and 'chat' in config.app:
                app_name = config.app.lower()[:config.app.lower().index('(')]
                config.engine = APP[app_name].engines()[0]

        '''
        when cfg has the same model_type in multi-sections, we use model_name + section_name to distinguish them.
        if has different model_type in multi-sections, we just use model_name to distinguish them.
        '''
        if len(set(model_types)) < len(configs) and len(configs) > 1:
            for config_name, config in configs.items():
                config.__setattr__('ir_prefix', config_name)

    def __call__(self):
        configs = copy.deepcopy(self.config_maps)
        # common_config = configs['Common']

        if len(configs) > 1:
            configs.pop("Common")

        self._config_preprocess(configs)
        build_pipeline = BuildPipeLine(configs)
        aipu_configs = build_pipeline()

        app_pipeline = AppPipeLine(aipu_configs)
        app_pipeline.forward()
