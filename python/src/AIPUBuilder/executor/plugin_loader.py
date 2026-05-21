from AIPUBuilder.logger import DEBUG, INFO, WARN
from AIPUBuilder.plugin_loader import *

EXECUTOR_DATASET_DICT = dict()

DM_PLUGINS = {t: dict() for t in PluginType}
DM_VERSIONS = {t: dict() for t in PluginType}


def register_plugin(type, version=0):
    global DM_PLUGINS, DM_VERSIONS

    def tofloat(str_version):
        try:
            fval = float(str_version)
        except:
            split_version = str_version.strip(' ').split('.')
            fval = float('.'.join([split_version[0], ''.join(split_version[1:])]))
        return fval

    def wrapper(cls):
        if type == PluginType.Dataset:
            dataset_name = cls.__name__.lower()
            if dataset_name not in EXECUTOR_DATASET_DICT:
                EXECUTOR_DATASET_DICT[dataset_name] = cls
            else:
                prev_version = tofloat(DM_VERSIONS[type][cls.__name__])
                cur_version = tofloat(version)
                if prev_version < cur_version:
                    DM_VERSIONS[type][cls.__name__] = str(version)
                    EXECUTOR_DATASET_DICT[dataset_name] = cls
                    (WARN(
                        'this dataset plugin(version=%s) %s has existed, and Executor will use the higher version(%s) to replace.'
                        % (str(prev_version), dataset_name, str(version))))
        if type not in DM_PLUGINS:
            DM_PLUGINS[type] = dict()
            DM_VERSIONS[type] = dict()
        if cls.__name__ not in DM_PLUGINS[type]:
            DM_PLUGINS[type][cls.__name__] = cls
            DM_VERSIONS[type][cls.__name__] = str(version)
        return cls

    return wrapper
