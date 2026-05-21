from typing import Any
from AIPUBuilder._C._tongue.ffc import *
from AIPUBuilder.core.utils import continue_class


@continue_class
class AIFFRegs:

    def __getattr__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{__name}'")


@continue_class
class AIFFGroupRegs:

    def __getattr__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{__name}'")


@continue_class
class Register:

    def __getattr__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{__name}'")
