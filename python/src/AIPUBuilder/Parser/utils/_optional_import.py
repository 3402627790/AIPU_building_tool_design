# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.

import importlib

_IMPORT_CACHE = {}


def optional_import(module_name):
    if module_name not in _IMPORT_CACHE:
        try:
            _IMPORT_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError:
            _IMPORT_CACHE[module_name] = None
    return _IMPORT_CACHE[module_name]


def require_optional_import(module_name, feature_name='this feature'):
    module_obj = optional_import(module_name)
    if module_obj is None:
        raise ImportError(
            'Missing optional dependency %s required for %s. Please install %s and retry.'
            % (module_name, feature_name, module_name)
        )
    return module_obj


class LazyOptionalModule(object):
    def __init__(self, module_name, feature_name='this feature'):
        self._module_name = module_name
        self._feature_name = feature_name
        self._module_obj = None

    def _load(self):
        if self._module_obj is None:
            self._module_obj = require_optional_import(self._module_name, self._feature_name)
        return self._module_obj

    def __getattr__(self, item):
        return getattr(self._load(), item)


def lazy_optional_import(module_name, feature_name='this feature'):
    return LazyOptionalModule(module_name, feature_name)
