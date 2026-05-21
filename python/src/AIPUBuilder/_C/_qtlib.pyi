from AIPUBuilder.core import Node

class Granularity(Enum):
    '''
    Granularity of tensor quantize.
    '''
    Per_Tensor = ...
    Per_Group = ...
    Per_Channel = ...


class Scheme(Enum):
    '''
    Scheme of tensor on AIPU.
    '''
    Symmetric = ...
    Asymmetric = ...


class QuantizedRange(Enum):
    '''
    QuantizedRange of tensor quantize.
    '''
    Full_Range = ...
    Restricted_Range = ...


class QuantizationTransform:
    def transform(self) -> None:
        """! dequantize one quantization format to float format
        """
        ...

    def quantize(self) -> None:
        """! quantize the float graph to aipu quantized graph
        """
        ...

    def forward(self) -> None:
        """! graph forward
        """
        ...

    def run(self) -> None:
        """! firstly transform and then quantize
        """
        ...

class Quantizer(object):
    def quantize(self) -> None:
        """! quantize the float graph to aipu quantized graph
        """
        ...