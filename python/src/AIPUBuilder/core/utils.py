#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3
import sys
from AIPUBuilder.utils import continue_class, wrap_class_method, RaiseCRuntimeError, DisableLogger


class BackupPassmgr(object):

    def __init__(self):
        self.disables = []

    def __enter__(self):
        from AIPUBuilder._C._core import get_disable_passes, enable_pass, disable_pass
        self.disables = get_disable_passes()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from AIPUBuilder._C._core import get_disable_passes, enable_pass, disable_pass

        for p in get_disable_passes():
            if p not in self.disables:
                enable_pass(p)
        for p in self.disables:
            disable_pass(p)
