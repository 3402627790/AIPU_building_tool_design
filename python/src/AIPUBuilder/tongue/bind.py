

from AIPUBuilder.core import *  # noqa
from AIPUBuilder.plugin_loader import *  # noqa
from AIPUBuilder.core.utils import DisableLogger
import numpy as np
import re

_C_keywords = [
    "alignas",  # C23
    "alignof",  # C23
    "auto",
    "bool",  # C23
    "break",
    "case",
    "char",
    "const",
    "constexpr",  # C23
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "false",  # C23
    "float",
    "for",
    "goto",
    "if",
    "inline",  # C99
    "int",
    "long",
    "nullptr",  # C23
    "register",
    "restrict",  # C99
    "return",
    "short",
    "signed",

    "sizeof",
    "static",
    "static_assert",  # C23
    "struct",
    "switch",
    "thread_local",  # C23
    "true",  # C23
    "typedef",
    "typeof",  # C23
    "typeof_unqual",  # C23
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Alignas",  # C11(deprecated in C23)
    "_Alignof",  # C11(deprecated in C23)
    "_Atomic",  # C11
    "_BitInt",  # C23
    "_Bool",  # C99(deprecated in C23)
    "_Complex",  # C99
    "_Decimal128",  # C23
    "_Decimal32",  # C23
    "_Decimal64",  # C23
    "_Generic",  # C11
    "_Imaginary",  # C99
    "_Noreturn",  # C11(deprecated in C23)
    "_Static_assert",  # C11(deprecated in C23)
    "_Thread_local",  # C11(deprecated in C23)
]


def _remove_comments(src):
    block_comments = r"/\*[\w\W]*?\*/"
    line_comments = r"//.*"
    return re.sub(block_comments + "|" + line_comments, "", src)


CLTYPES = set([
    "%s%d" % (base_name, count)
    for base_name in [
        'bool',
        'char', 'uchar', 'short', 'ushort', 'int', 'uint',
        'long', 'ulong', 'float', 'double', 'half']
    for count in [2, 3, 4, 8, 16, 32]
])


def my_cl_func_parser(src):
    import re
    src = _remove_comments(src)
    idp = r"[a-zA-Z_][a-zA-Z0-9_]*"
    args = r"\s*\(\s*([0-9a-zA-Z_,\*\s]+)\s*\)\s*"
    kern = r"(?!\n)\s*__kernel\s+" + idp + r"\s*\s*(%s)\s*\s*" % idp + args
    ret = []
    for f in re.findall(kern, src):
        name, args = f
        func = CLFunction(name)
        args = args.split(",")
        for a in args:
            a = a.strip()
            b = re.split(r"\s+", a)
            is_ptr = False
            name = ""
            types = []
            if b[-1].endswith("*"):
                is_ptr = True
                b[-1] = b[-1][:-1]
                types = b[:-1]
            elif (len(b) > 1 and b[-2].endswith("*")):
                is_ptr = True
                types = b[:-2]
                name = b[-1][1:]
            elif b[-1].startswith("*"):
                is_ptr = True
                types = b[:-1]
                name = b[-1][1:]
            else:
                if b[-1] not in CLTYPES:
                    types = b[:-1]
                    name = b[-1]
            ar = CLFunction.Argument(name, is_ptr, types, types)
            func.args.append(ar)
        ret.append(func)
    return ret

# OpenCLCParser.initial_type_symbols |= set([
#     "%s%d" % (base_name, count)
#     for base_name in [
#         'bool',
#         'char', 'uchar', 'short', 'ushort', 'int', 'uint',
#         'long', 'ulong', 'float', 'double', 'half']
#     for count in [2, 3, 4, 8, 16, 32]
# ])


_buildinoptype = register_optype("_BuildInClKernelBind")


@register_plugin(PluginType.Builder)
class BuildinClKernelBindPlugin(BuilderOpPlugin):
    CFLAG = ["-g", "-O2"]

    def get_graph_pattern(self):
        return ([("0", _buildinoptype)], [])

    def setup(self, sgnode, nodes):
        sgnode.attrs["keeping_layout"] = False
        valid_in = []
        valid_out = []
        sgnode.attrs["valid_input_layouts"] = valid_in
        sgnode.attrs["valid_output_layouts"] = valid_out
        return True

    def generate_params(self, sgnode, nodes):
        ro = BuilderParams()
        for i in range(60):
            i = str(i)
            if i not in nodes[0].params:
                break
            v = nodes[0].params[i]
            if isinstance(v, str):
                i = int(v[1:])
                if "i" == v[0]:
                    ro.append(sgnode.inputs[i])
                elif "o" == v[0]:
                    ro.append(sgnode.outputs[i])
                else:
                    raise Exception("unsupport value")

            else:
                ro.append(v)
        return ro

    def generate_code(self, sgnode, nodes):
        return nodes[0].params["src"]

    def generate_kernel(self, sgnode, nodes):
        if "kernel" in nodes[0].params:
            return nodes[0].params["kernel"]
        return super().generate_kernel(sgnode, nodes)


class CLFunction(object):
    class Argument(object):
        def __init__(self, name="", ptr=False, quals=[], types=[]) -> None:
            self.name = name
            self.ptr = ptr
            self.quals = quals
            self.types = types
            # if self.quals == self.types:
            #     self.quals=[]

        def __str__(self):
            ptr = ""
            if self.ptr:
                ptr = "*"
            quals = self.quals
            if self.quals == self.types:
                quals = []
            return " ".join(quals) + " " + " ".join(self.types) + ptr + " " + self.name

    def __init__(self, fname: str) -> None:
        self.name = fname
        self.args = []

    def __str__(self):
        return f"{self.name}({','.join([str(i) for i in self.args])})"


def _bind_func(f: CLFunction, src: str, *bargs, **kwargs):
    node = Node("test", _buildinoptype)

    def bf(*args):
        # print( f"expected args: {str(f)} but got {','.join([str(i) for i in args])}")
        if len(args) != len(f.args):
            raise Exception(
                f"expected args: {str(f)} but got {','.join([str(i) for i in args])}")
        node.params["src"] = src
        node.params["kernel"] = f.name
        node.clear_inputs()

        node.clear_outputs()
        has_tensor = False
        for i, v in enumerate(args):
            arg = f.args[i]
            if isinstance(v, np.ndarray) and "__global" in arg.quals:
                node.add_input(Tensor(v))
                node.params[str(i)] = "i" + str(len(node.inputs) - 1)
                has_tensor = True
            elif isinstance(v, Tensor) and "__global" in arg.quals:
                if v.data() is not None and "const" in arg.quals:
                    node.add_input(v)
                    node.params[str(i)] = "i" + str(len(node.inputs) - 1)
                elif "const" in arg.quals:
                    raise Exception(
                        f"{str(arg)} is expected as an input, but call with a Tensor without data.")
                else:
                    node.add_output(v)
                    node.params[str(i)] = "o" + str(len(node.outputs) - 1)
                has_tensor = True
            elif "__global" in arg.quals:
                raise Exception(
                    f"{str(arg)} only support numby array or Tensor, but got {type(v)}")
            elif isinstance(v, Tensor):
                node.add_output(Tensor(v))
                node.params[str(i)] = "o" + str(len(node.outputs) - 1)
                has_tensor = True
            else:
                node.params[str(i)] = v
        if not has_tensor:
            raise ValueError(
                "the kernel function has not 'Tensor' arguments, which will not be build, if you are using scalar as paramerters, please wrapper in a Tensor obj.")
        with DisableLogger():
            # if True:
            if "_elf" in node.attrs and node.attrs["_elf"] is not None:
                # rewrite ro
                elf = node.attrs["_elf"]
                for i in range(60):
                    si = str(i)
                    if si not in node.params:
                        break
                    v = node.params[si]
                    if isinstance(v, str):
                        i = int(v[1:])
                        continue
                    else:
                        elf.rewrite_ro(i+64, v)
            run_node(node, *bargs, **kwargs)

        if len(node.outputs) == 1:
            return node.outputs[0].numpy()
        return tuple([o.numpy() for o in node.outputs])
    bf._meta = f
    bf._node = node
    return bf


def bind(func, *args, **kwargs):
    # parser = OpenCLCParser()
    # ast = parser.parse(func)
    # visitor = ExtFuncVisitor()
    # visitor.visit(ast)
    # f = visitor.ext_funcs[0]
    f = my_cl_func_parser(func)[0]
    return _bind_func(f, func, *args, **kwargs)


def _import_cl_file(src, *args, **kwargs):
    # parser = OpenCLCParser()
    # ast = parser.parse(src)
    # visitor = ExtFuncVisitor()
    # visitor.visit(ast)
    frame = sys._getframe(2)
    ret = []
    # for f in visitor.ext_funcs:
    for f in my_cl_func_parser(src):
        ret.append(_bind_func(f, src, *args, **kwargs))
        frame.f_globals[f.name] = ret[-1]
    return ret


def import_cl(path, *args, **kwargs):
    with open(path, "r") as f:
        return _import_cl_file(f.read(), *args, **kwargs)
