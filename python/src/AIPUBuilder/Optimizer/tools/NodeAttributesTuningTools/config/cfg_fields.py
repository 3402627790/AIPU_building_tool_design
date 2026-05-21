import os

DEFAULT_FIELDS = {}
HIDDEN_FIELDS = {}


def field_register(field, ftype='default'):
    def wrapper(cls):
        if ftype.lower() == 'default':
            DEFAULT_FIELDS.update({field: cls})

        if ftype.lower() == 'hidden':
            HIDDEN_FIELDS.update({field: cls})
        return cls
    return wrapper


class BaseField(object):
    @staticmethod
    def default(*args):
        raise NotImplementedError()

    @staticmethod
    def check(*args):
        raise NotImplementedError()

    @staticmethod
    def error(*args):
        raise NotImplementedError()

    @staticmethod
    def message(*args):
        raise NotImplementedError()


@field_register('graph', 'default')
class GraphField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(g):
        return True if os.path.isfile(g) else False

    @staticmethod
    def error(g):
        return f"Require existed graph path, now graph path is {g}."

    @staticmethod
    def message():
        return "Graph description file path of unquantized Compass IR."


@field_register('bin', 'default')
class BinField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(b):
        return True if os.path.isfile(b) else False

    @staticmethod
    def error(b):
        return f"Require existed bin path, now graph path is {b}."

    @staticmethod
    def message():
        return "Binary file path of unquantized Compass IR."


@field_register('dump_dir', 'default')
class OutputDirField(BaseField):
    # path to save the optimized IR and other log files
    @staticmethod
    def default():
        return './dump'

    @staticmethod
    def check(od):
        return True if os.path.exists(od) else False

    @staticmethod
    def error(od):
        return f"Now 'dump_dir={od}' does not existed. Optimizer will create the path."

    @staticmethod
    def message():
        return f"Path to store generated IR, and other output files"


@field_register('attr_entry', 'hidden')
class AttrEntryField(BaseField):
    entry_list = ["hparam"]

    @staticmethod
    def default():
        return 'hparam'

    @staticmethod
    def check(v):
        return True if v in AttrEntryField.entry_list else False

    @staticmethod
    def error(v):
        return f"Unknown entry point {v}. Valid entries are {AttrEntryField.entry_list}"

    @staticmethod
    def message():
        return f"Position to inject attributes, such as hparam in Compass OPT"


@field_register('attrs', 'hidden')
class AttrsField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(v):
        return isinstance(v, dict)

    @staticmethod
    def error(v):
        return "Cannot parse attrs dict"

    @staticmethod
    def message():
        return "Name of attributes and potential values: E.x. {'attr1':[0,1,2],'attr2':[3,4]}"


@field_register('backend', 'default')
class BackendField(BaseField):
    valid = ["opt", "simulator", "rpc", "emulator"]

    @staticmethod
    def default():
        return ["opt"]

    @staticmethod
    def check(v):
        for term in v:
            if term not in BackendField.valid:
                return False
        return True

    @staticmethod
    def error(v):
        return f"{v} is not valid for backend list. Supported backends are {BackendField.valid}"

    @staticmethod
    def message():
        return f"Select execute backends"


@field_register('target', 'default')
class TargetField(BaseField):
    @staticmethod
    def default():
        return "X1_1204"

    @staticmethod
    def check(v):
        return True

    @staticmethod
    def error(v):
        return ""

    @staticmethod
    def message():
        return f"Target for simulator, defualt is X1_1204"


@field_register('simulator', 'default')
class SimulatorField(BaseField):
    @staticmethod
    def default():
        return "/project/ai/scratch01/AIPU_SIMULATOR/aipu_simulator_x1"

    @staticmethod
    def check(v):
        return True

    @staticmethod
    def error(v):
        return ""

    @staticmethod
    def message():
        return f"Simulator path"


@field_register('gt', 'default')
class GtField(BaseField):
    @staticmethod
    def default():
        return "/project/ai/scratch01/GtGenerator/GtGenerator"

    @staticmethod
    def check(v):
        return True

    @staticmethod
    def error(v):
        return ""

    @staticmethod
    def message():
        return f"GtGenerator path"


@field_register('simulator_option', 'hidden')
class SimulatorOptionField(BaseField):
    @staticmethod
    def default():
        return ""

    @staticmethod
    def check(v):
        return isinstance(v, str)

    @staticmethod
    def error(v):
        return "option should be string"

    @staticmethod
    def message():
        return "plugin option of aipurun, will append to aipurun cmd line"


@field_register('opt_option', 'hidden')
class OPTOptionField(BaseField):
    @staticmethod
    def default():
        return ""

    @staticmethod
    def check(v):
        return isinstance(v, str)

    @staticmethod
    def error(v):
        return "option should be string"

    @staticmethod
    def message():
        return "option of opt forward, such as transfer result to float (true or false)"


@field_register('quantizer_hparams', 'hidden')
class QuantizerHparamsField(BaseField):
    @staticmethod
    def default():
        return dict()

    @staticmethod
    def check(v):
        return isinstance(v, dict)

    @staticmethod
    def error(v):
        return "Hparams should be stored in a map"

    @staticmethod
    def message():
        return "Provide extra hyper params to quantizer. Please run 'aipuopt -f' for details"
