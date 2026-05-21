from typing import List
from enum import Enum
from AIPUBuilder.core import Tensor, NodeParamValue


class ISAType(Enum):
    """An enumeration of different ISA types."""
    ZY = ...
    ZYv2 = ...
    ZYv3 = ...
    ZYv5 = ...
    ZYv6 = ...


class Target:
    '''
    A class for Build target
    '''

    def __init__(self) -> None: ...
    @property
    def name(self) -> str:
        """Returns the name of the target."""
    @property
    def isa(self) -> ISAType:
        """Returns the ISA type of the target."""
    @property
    def tec_num(self) -> int:
        """Returns the number of TECs in the target."""
    @property
    def core_num(self) -> int:
        """Returns the number of cores in the target."""
    @property
    def gsram_size(self) -> int:
        """Returns the size of global SRAM in the target."""
    @property
    def lsram_size(self) -> int:
        """Returns the size of local SRAM in the target."""
    @property
    def ff_mac_num(self) -> int:
        """Returns the number of FF MACs in the target."""
    @classmethod
    def set_target(self, target: str) -> None:
        """Sets the current build target."""


class ReadOnlyParam:
    '''
    a class for building read only parameter
    The read only parameter only accept a int32 or Tensor type, where int32 is a 32bit width value and Tensor will be a memory pointer of the Tensor data.
    '''
    value: int
    tensor: Tensor
    def __init__(self, v: [int, Tensor, ReadOnlyParam, NodeParamValue]) -> None: ...


class BuilderParams(list):
    def append(self, ro: [int, Tensor, ReadOnlyParam, NodeParamValue]) -> None: ...
