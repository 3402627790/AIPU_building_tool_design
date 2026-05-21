from AIPUBuilder.logger import ERROR
from AIPUBuilder.model.register import register_engine
from .baseengine import BaseEngine


@register_engine('opt')
class OptEngine(BaseEngine):
    def __init__(self, config=None, irt='', irb=''):

        from AIPUBuilder.Optimizer.tools.optimizer_forward import OptForward
        if config is None and irt == '' and irb == '':
            ERROR("OptEngine must have a correct config or ir_txt and ir_bin.")

        self.config = config
        if self.config is not None:
            self.irt = config.get('quantized_ir_path', '')
            if config.get('gsim_ir_path', '') != '':
                self.irt = config.gsim_ir_path
            if self.irt == '' and config.get('float_ir_path', '') != '':
                self.irt = config.float_ir_path
            self.irb = self.irt.replace('.txt', '.bin')
        elif irt != '':
            self.irt = irt
            self.irb = irb

        self.opt_engine = OptForward(self.irt, self.irb)
        if self.opt_engine.graph is None:
            ERROR("OptEngine must have a Graph, not None.")

    @property
    def input_shapes(self):
        return [list(inp.ir_shape) for inp in self.opt_engine.graph.input_tensors]

    @property
    def input_dtypes(self):
        return [t.dtype.name for t in self.opt_engine.graph.input_tensors]

    @staticmethod
    def engine_name():
        return 'opt'

    def forward(self, inputs, **kwargs):
        return self.opt_engine.forward(inputs, **kwargs)
