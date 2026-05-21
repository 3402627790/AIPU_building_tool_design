import torch

from AIPUBuilder._C._core import DataLayout

norm2non = {
    DataLayout.Flat: lambda x: x,

    DataLayout.NHWC:      lambda x: x,
    DataLayout.NCHW:      lambda x: torch.permute(x, [0, 3, 1, 2]),
    DataLayout.NCHWC16:   lambda x: norm2align(x, 16),
    DataLayout.NCHWC16T1: lambda x: norm2align_tile(x, 16),
    DataLayout.NCHWC16T8: lambda x: norm2align_tile(x, 16),
    DataLayout.NCHWC32:   lambda x: norm2align(x, 32),
    DataLayout.NCHWC32T1: lambda x: norm2align_tile(x, 32),
    DataLayout.NCHWC32T8: lambda x: norm2align_tile(x, 32),

    DataLayout.NDHWC:      lambda x: x,
    DataLayout.NDCHWC16:   lambda x: norm2align(x, 16),
    DataLayout.NDCHWC16T1: lambda x: norm2align_tile(x, 16),
    DataLayout.NDCHWC16T8: lambda x: norm2align_tile(x, 16),
    DataLayout.NDCHWC32:   lambda x: norm2align(x, 32),
    DataLayout.NDCHWC32T1: lambda x: norm2align_tile(x, 32),
    DataLayout.NDCHWC32T8: lambda x: norm2align_tile(x, 32),
}

non2norm = {
    DataLayout.Flat: lambda x: x,

    DataLayout.NHWC:      lambda x: x,
    DataLayout.NCHW:      lambda x: torch.permute(x, [0, 2, 3, 1]),
    DataLayout.NCHWC16:   lambda x: align2norm(x, 16),
    DataLayout.NCHWC16T1: lambda x: align_tile2norm(x, 16),
    DataLayout.NCHWC16T8: lambda x: align_tile2norm(x, 16),
    DataLayout.NCHWC32:   lambda x: align2norm(x, 32),
    DataLayout.NCHWC32T1: lambda x: align_tile2norm(x, 32),
    DataLayout.NCHWC32T8: lambda x: align_tile2norm(x, 32),

    DataLayout.NDHWC:      lambda x: x,
    DataLayout.NDCHWC16:   lambda x: align2norm(x, 16),
    DataLayout.NDCHWC16T1: lambda x: align_tile2norm(x, 16),
    DataLayout.NDCHWC16T8: lambda x: align_tile2norm(x, 16),
    DataLayout.NDCHWC32:   lambda x: align2norm(x, 32),
    DataLayout.NDCHWC32T1: lambda x: align_tile2norm(x, 32),
    DataLayout.NDCHWC32T8: lambda x: align_tile2norm(x, 32),
}


def norm2align(x, align):
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1]//align, align)
    x = x.transpose(-2, -3)
    x = x.transpose(-3, -4)
    return x


def norm2align_tile(x, align):
    shape = x.shape
    c, tile = divmod(shape[-1], align)
    x1, x2 = torch.split(x, [c * align, tile], dim=-1)
    x1 = norm2align(x1, align)
    x1 = x1.reshape(*shape[:-3], -1)
    x2 = x2.reshape(*shape[:-3], -1)
    x = torch.cat([x1, x2], dim=-1)
    return x.reshape(shape)


def align2norm(x, align):
    x = x.transpose(-4, -3)
    x = x.transpose(-3, -2)
    x = x.reshape(*x.shape[:-2], -1)
    return x


def align_tile2norm(x, align):
    shape = x.shape
    h, w, c = shape[-3:]
    c_num, tile = divmod(c, align)
    x = x.reshape(*shape[:-3], -1)
    x1, x2 = torch.split(x, [w * h * align * c_num, w * h * tile], dim=-1)
    x1 = x1.reshape(*shape[:-3], c_num, h, w, align)
    x1 = align2norm(x1, align)
    x2 = x2.reshape(*shape[:-3], h, w, tile)
    return torch.cat([x1, x2], dim=-1)

