import threading
from att_utils.logger import INFO, WARN

SUPPORTED_BACKENDS = {}


def backend_register(field):
    def wrapper(cls):
        SUPPORTED_BACKENDS.update({field: cls})
        return cls
    return wrapper


class BaseBackend:

    def __init__(self, databus, IR_graph, IR_bin):
        self.IR_graph = IR_graph
        self.IR_bin = IR_bin
        self.databus = databus

    def set_metric(self):
        raise NotImplementedError

    def prepare_input(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError


class Backend:

    def __init__(self, databus, name):
        self.databus = databus
        self.name = name
        self.backends = []

    def add_backend(self, field, IR_graph, IR_bin):
        backend = SUPPORTED_BACKENDS[field](self.databus, IR_graph, IR_bin,
                                            batch=self.databus.backend_dataset.data_batch)
        backend.prepare_input()
        backend.set_metric()
        self.backends.append([field, backend])

    def execute(self):
        metrics = {}
        threads = []
        for field, backend in self.backends:
            def run_process():
                metrics[field] = backend.execute()
                INFO(f"Backend:{self.name} {field} metric:", metrics[field])
            thread = threading.Thread(target=run_process)
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()
        return metrics
