import importlib
import pkgutil
import functools

from enum import Enum, unique, IntFlag
import os
import sys
from AIPUBuilder.logger import WARN
AIPUPLUGIN_PATH = os.environ.get("AIPUPLUGIN_PATH", "").split(":")

PLUGIN_PREFIX = "aipubt_"


class PluginTypeError(Exception):
    pass


@unique
class PluginType(IntFlag):
    Dataset = 0x110
    Builder = 0x314
    Checker = 0x315
    Parser = 0x200
    PreGSIM = 0x201
    PostGSIM = 0x202


@unique
class PluginFlag(IntFlag):
    GBPlugin_DS = 0x1 << 16


PLUGINS = {t.value: dict() for t in PluginType}
VERSIONS = {t.value: dict() for t in PluginType}
PLUGIN_FILES = set()


def register_plugin(plugin_type, version=0):
    '''
    register a plugin with plugin_type and version
    supporting plugin plugin_type: Parser,Builder
    '''
    import inspect
    global PLUGINS, VERSIONS, PLUGIN_FILES

    frame = inspect.stack()[1]
    p = frame[0].f_code.co_filename
    PLUGIN_FILES.add(os.path.realpath(p))
    flag = plugin_type.value >> 16
    plugin_type = PluginType(plugin_type.value & 0xfff)

    def tofloat(str_version):
        try:
            fval = float(str_version)
        except:
            split_version = str_version.strip(' ').split('.')
            fval = float(
                '.'.join([split_version[0], ''.join(split_version[1:])]))
        return fval

    def wrapper(cls):
        if plugin_type not in PluginType:
            raise PluginTypeError("Unsupported Plugin Type.")
        elif plugin_type == PluginType.Parser:
            from AIPUBuilder.Parser.plugin_loader import register_plugin as register_parser_plugin
            register_parser_plugin(version=version)(cls)
        elif plugin_type == PluginType.Builder:
            from AIPUBuilder.CGBuilder._plugin_loader import register_builder_plugin
            # process version
            register_builder_plugin(version, flag)(cls)
            pass
        elif plugin_type == PluginType.Checker:
            from AIPUBuilder.CGBuilder._plugin_loader import register_checker_plugin
            # process version
            register_checker_plugin(version)(cls)
            pass
        elif plugin_type == PluginType.Dataset:
            from AIPUBuilder.executor.plugin_loader import register_plugin as register_dataset_plugin
            register_dataset_plugin(PluginType.Dataset, version=version)(cls)
        elif plugin_type == PluginType.PreGSIM:
            from AIPUBuilder.simplifier.plugin import register_pre_pass, register_pass_function, BasePassPlugin
            if issubclass(cls, BasePassPlugin):
                register_pre_pass(cls)
            else:
                register_pass_function(register_pre_pass, cls, version)
        elif plugin_type == PluginType.PostGSIM:
            from AIPUBuilder.simplifier.plugin import register_post_pass, register_pass_function, BasePassPlugin
            if issubclass(cls, BasePassPlugin):
                register_post_pass(cls)
            else:
                register_pass_function(register_post_pass, cls, version)
        if plugin_type not in PLUGINS:
            PLUGINS[plugin_type] = dict()
            VERSIONS[plugin_type] = dict()
        if cls.__name__ not in PLUGINS[plugin_type]:
            PLUGINS[plugin_type][cls.__name__] = cls
            VERSIONS[plugin_type][cls.__name__] = str(version)
        return cls
    return wrapper


AIPUPLUGIN_PATH = [i for i in set(
    AIPUPLUGIN_PATH + ["./plugin", "."]) if len(i) != 0]
sys.path = AIPUPLUGIN_PATH + sys.path

ALL_MODULES = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in pkgutil.iter_modules(AIPUPLUGIN_PATH)
    if name.startswith(PLUGIN_PREFIX)
}
