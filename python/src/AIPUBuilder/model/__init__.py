from .basemodel import *
from .config import *
from .qinfo import *
from .utils import *
from .modelbuild import ModelBuild
from .register import *
from .aipullm import AIPULLM
from .llm import *
from .pipeline import *
from .loadweight import *
from .app import *
from .engines import *

import importlib
import pkgutil
import sys
import os
PLUGIN_PREFIX = "aipubt_"
AIPUPLUGIN_PATH = os.environ.get("AIPUPLUGIN_PATH", "").split(":")
AIPUPLUGIN_PATH = [i for i in set(AIPUPLUGIN_PATH+["./plugin", "."]) if len(i) != 0]
sys.path = AIPUPLUGIN_PATH+sys.path

_MODULES = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in pkgutil.iter_modules(AIPUPLUGIN_PATH)
    if name.startswith(PLUGIN_PREFIX)
}
