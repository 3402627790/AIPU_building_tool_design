from AIPUBuilder._C._builder import _py_builderop_register, BuilderParams
from AIPUBuilder._C._builder import BuilderOpPluginInfo
from AIPUBuilder._C._checker import _py_checkerop_register
from AIPUBuilder._C._checker import CheckerOpPluginInfo
from AIPUBuilder._C._builder import cache_text_content
from AIPUBuilder.core import API_VERSION


def register_builder_plugin(version, flag=0):
    def wrapper(cls):
        p = BuilderOpPluginInfo()
        p.api_version = API_VERSION
        p.plugin_version = int(version)
        p.plugin_type = (0x314 & 0xffff) | (flag << 16)
        p.name = cls.__name__
        if not hasattr(cls, "generate_params"):
            raise Exception("The Plugin")
        _gen_ro = cls.generate_params
        _gen_desc = cls.generate_descriptor
        p.cls = cls
        # wrapper cls gen ro and gen desc

        def gen_ro(*args):
            ret = _gen_ro(*args)
            if not isinstance(ret, BuilderParams):
                from AIPUBuilder.logger import FATAL
                FATAL(
                    f"The plugin {cls.__name__} 'generate_params' must return a BuilderParams type, but got a type {type(ret)}")
            return ret
        cls.generate_params = gen_ro

        def gen_desc(*args):
            ret = _gen_desc(*args)
            if not isinstance(ret, BuilderParams):
                from AIPUBuilder.logger import FATAL
                FATAL(
                    f"The plugin {cls.__name__} 'generate_descriptor' must return a BuilderParams type, but got a type {type(ret)}")
            return ret
        cls.generate_descriptor = gen_desc
        _py_builderop_register(p)
        return cls
    return wrapper


def register_checker_plugin(version):
    def wrapper(cls):
        p = CheckerOpPluginInfo()
        p.api_version = API_VERSION
        p.plugin_version = int(version)
        p.plugin_type = 0x315
        p.name = cls.__name__
        p.cls = cls
        _py_checkerop_register(p)
        return cls
    return wrapper
