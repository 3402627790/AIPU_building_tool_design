_configs = {}

def set_configs(conf: dict):
    _configs.clear()
    _configs.update(conf)

def get_config(key, default=None):
    return _configs.get(key, default)

def get_all_configs():
    return _configs.copy()