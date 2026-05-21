import json
import os
from .cfg_fields import DEFAULT_FIELDS, HIDDEN_FIELDS
from plugins.plugin import DATASET_PLUGINS, METRIC_PLUGINS, STRATEGY_PLUGINS
from AIPUBuilder.Optimizer.framework.opt_register import QUANTIZE_METRIC_DICT
from att_utils.logger import INFO, WARN


class Parser:

    def __init__(self, databus):
        self.databus = databus
        for k, cls in METRIC_PLUGINS.items():
            QUANTIZE_METRIC_DICT.update({k: cls})

    def show_fields(self):
        os_str = 'Now Optimizer supports configurate parameters in .json file:\n'
        show_dict = {}
        for key, val in DEFAULT_FIELDS.items():
            show_dict.update({key: val.message()})
        for key, val in HIDDEN_FIELDS.items():
            show_dict.update({key: val.message()})

        max_len = max([len(k) for k in show_dict.keys()])
        for k, v in show_dict.items():
            os_str += ('\t--{: <%d}  {: <}\n' % max_len).format(k, v)
        return os_str

    def show_plugins(self):
        os_str = 'Now valid dataset, metric and strategy plugins are:\n'
        show_dict = {}
        for key, val in DATASET_PLUGINS.items():
            show_dict.update({key: val.message()})
        for key, val in QUANTIZE_METRIC_DICT.items():
            show_dict.update({key: "OPT_METRIC"})
        for key, val in STRATEGY_PLUGINS.items():
            show_dict.update({key: val.message()})

        max_len = max([len(k) for k in show_dict.keys()])
        for k, v in show_dict.items():
            os_str += ('\t--{: <%d}  {: <}\n' % max_len).format(k, v)
        return os_str

    def load(self):
        path_args = ["graph", "bin", "dataset", "dump_dir"]
        with open(self.databus["cfg_path"], 'r') as f:
            data = json.load(f)
        # load mandatory params
        for name, field in DEFAULT_FIELDS.items():
            val = field.default()
            if name in data.keys():
                val = data[name]
            if name in path_args and os.path.isfile(val):
                val = os.path.join(self.databus["base_path"], val)
            if not field.check(val):
                WARN(field.error(val))
            self.databus[name] = val
        # load optional params
        for name, value in data.items():
            if name in HIDDEN_FIELDS.keys():
                if not HIDDEN_FIELDS[name].check(value):
                    WARN(HIDDEN_FIELDS[name].error())
                self.databus[name] = value
        # load dataset
        if not "dataset" in data.keys():
            WARN("Backend may need dataset to run metrics")
        else:
            dataset = DATASET_PLUGINS[data["dataset"][0]]()
            argv = data["dataset"][1:]
            dataset.prepare(argv)
            self.databus["backend_dataset"] = dataset
        # load metric
        if not "metric" in data.keys():
            WARN("Strategy may need metric to evaluate attributes")
        else:
            self.databus["backend_metric_class"] = QUANTIZE_METRIC_DICT[data["metric"][0]]
            self.databus["backend_metric_arg"] = data["metric"][1:]
        # load strategy
        assert "strategy" in data.keys(), "[FATAL] Need Strategy to organize work flow"
        strategy = STRATEGY_PLUGINS[data["strategy"]](self.databus)
        return strategy
