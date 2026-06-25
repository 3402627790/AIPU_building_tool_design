#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import os
__VERSION__ = "6.1"
__build_number__='dev20490'
if __build_number__ is not None:
    __VERSION__ = __VERSION__+"."+str(__build_number__)

__release__ = False

_basedir = os.path.dirname(os.path.abspath(__file__))
_C_dir = os.path.abspath(os.path.join(_basedir, "_C"))
_plugindir = os.path.abspath(os.path.join(_C_dir, "plugin"))
_env = _plugindir
_CACHE_DIR = f"/var/tmp/aipubuilder_{os.getuid()}"
if "AIPUPLUGIN_PATH" in os.environ:
    _env = os.environ["AIPUPLUGIN_PATH"]
    if _plugindir not in _env:
        _env = _env+":"+_plugindir
os.environ["AIPUPLUGIN_PATH"] = _env
_AIPULIB_PATH = _C_dir+f":{_C_dir}/lib"
if "AIPULIB_PATH" in os.environ:
    _AIPULIB_PATH = os.environ["AIPULIB_PATH"]
    if _C_dir not in _AIPULIB_PATH:
        _AIPULIB_PATH = _AIPULIB_PATH+":"+_C_dir
    if f"{_C_dir}/lib" not in _AIPULIB_PATH:
        _AIPULIB_PATH = _AIPULIB_PATH+":"+f"{_C_dir}/lib"
_AIPULIB_PATH += ":"+_CACHE_DIR+":."
os.environ["AIPULIB_PATH"] = _AIPULIB_PATH
