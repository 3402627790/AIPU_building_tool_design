from .plugin import BaseStrategy, strategy_plugin_register
from quantizer.qtlib import QtlibQuantizer
from quantizer.opt import OPTQuantizer
from backend.backend import Backend
from multiprocessing import Process, Queue
from threading import Thread
from itertools import product
from att_utils.logger import INFO, WARN
from att_utils.to_pd import convert_single_metric_to_pd
from IPython.display import display


@strategy_plugin_register("parallel_loop_strategy")
class ParallelLoopStrategy(BaseStrategy):

    def __init__(self, databus):
        self.databus = databus
        self.metric_results = {}
        INFO("Parallel Loop Strategy Initialized")

    def call_backend(self):
        IR = self.databus.quantized_IR[self.databus.quantize_frame].pop()
        attrs = IR["attr"]
        graph = IR["graph"]
        weight = IR["bin"]
        INFO("Calling Backend with attr", attrs)
        INFO("\tIR:", graph)
        backend = Backend(self.databus, str(attrs))
        for field in self.databus.backend:
            backend.add_backend(field, graph, weight)
            INFO(f"Initializing {field} with attr", attrs)
        metric = backend.execute()
        return [str(attrs), metric]

    def __call__(self):
        INFO("Calling for quantizer(s)")
        self.databus["quantized_IR"] = {
            "qtlib": [],
            "opt": []
        }
        # permutation of attributes
        keys, values = zip(*self.databus.attrs.items())
        pd = [dict(zip(keys, v)) for v in product(*values)]
        attr_list = []
        for combine in pd:
            attr = []
            for k, v in combine.items():
                attr.append((k, v))
            attr_list.append(attr)

        Quantizer = OPTQuantizer
        self.databus.quantize_frame = "opt"
        with open(self.databus.graph, 'r') as f:
            meta_count = 0
            for line in f.readlines():
                meta_count += 1
                line = line.lower()
                if "compat_quantized_model" in line and "true" in line:
                    Quantizer = QtlibQuantizer
                    self.databus.quantize_frame = "qtlib"
                    break
                if meta_count > 20:
                    break

        quantizers = []
        for attrs in attr_list:
            quantizer = Quantizer(self.databus, attrs)
            quantizers.append(quantizer)

        def run_quantizer(quantizer):
            INFO("Quantizing with attr", quantizer.attrs)
            quantizer.quantize()
            INFO("Quantizer finished. Attr:", quantizer.attrs)
            attr, metric = self.call_backend()
            self.metric_results.update({attr: metric})

        quantizer_process = []
        for q in quantizers:
            process = Thread(target=run_quantizer, args=[q])
            quantizer_process.append(process)

        for p in quantizer_process:
            p.start()
            INFO("Quantizer Process started")

        for p in quantizer_process:
            p.join()

        INFO("Searching complete. Metrics are")
        # for _ in range(len(quantizers)):
        #     attr, metric = queue.get()
        #     self.metric_results.update({attr:metric})

        pd = convert_single_metric_to_pd(self.metric_results)
        for bn, data in pd.items():
            INFO(f"Backend:{bn} has metrics:")
            display(data)
            data.to_csv(f"aipuatt_{bn}.csv")

    @staticmethod
    def message():
        return "The plugin will loop all potential attributes and evaluate them parallel. All results will be displayed"
