import os
import subprocess
import numpy as np
from att_utils.logger import INFO, WARN
from backend.backend import BaseBackend, backend_register


@backend_register("simulator")
class Simulator(BaseBackend):

    def __init__(self, databus, IR_graph, IR_bin, **kargs):
        super().__init__(databus, IR_graph, IR_bin)
        self.target = self.databus.target
        IR_name = os.path.basename(IR_graph)
        self.oup = os.path.join(self.databus['dump_dir'], IR_name.split('.')[0] + "_simulator_output_")
        self.simulator_path = self.databus.simulator
        self.batch_size = kargs["batch"] if "batch" in kargs.keys() else 1

    def prepare_input(self):
        data = self.databus.backend_dataset.load()  # load batch data
        path = os.path.join(self.databus.dump_dir, "simulator_input.bin")
        data.tofile(path)
        self.inp = path

    def set_metric(self):
        self.metric = self.databus.backend_metric_class()
        self.metric.prepare(self.databus.backend_metric_arg[0])

    def execute(self):
        log_file = self.oup[:-4] + ".log"
        option = ""
        if "simulator_option" in self.databus:
            option = self.databus.simulator_option
        inp = np.fromfile(self.inp, dtype=np.int8)
        batch = inp.shape[0] // self.batch_size
        outputs = []
        processes = []

        for i in range(self.batch_size):
            inp_batch = inp[batch * i:batch * (i+1)]
            inp_name = self.IR_graph + ".inp_"+str(i)+".bin"
            inp_batch.tofile(inp_name)
            cmd = f"aipurun {self.IR_graph} -w {self.IR_bin} -i {inp_name} -o {self.oup}{i}.bin --target {self.target} --simulator {self.simulator_path} {option} > {log_file} 2>&1"
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)

        for p in processes:
            p.wait()

        for i in range(self.batch_size):
            outputs.append(np.fromfile(f"{self.oup}{i}.bin", dtype=self.databus.simulator_output_type))

        INFO("Simulator done", self.IR_graph)
        res = np.array(outputs)
        return self.metric(res)
