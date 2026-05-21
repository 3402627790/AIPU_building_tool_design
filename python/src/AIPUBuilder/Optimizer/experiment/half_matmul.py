import torch

mask = torch.tensor(0xFFFFFF00).int()


def half_matmul(a: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:
    dtype = a.dtype
    a = (a.float().view(torch.int32) & mask).view(torch.float32)
    b = (b.float().view(torch.int32) & mask).view(torch.float32)

    assert a.dim() == b.dim()
    assert a.dim() >= 2, "Too few dim to call ADDMM"

    keep_shape = [d for d in a.shape][:-2]
    m, dm = a.shape[-2], a.shape[-1]
    dn, n = b.shape[-2], b.shape[-1]
    assert dm == dn, "Mat dim mismatch"
    d = dm = dn

    a = a.reshape(-1, m, d)
    b = b.reshape(-1, d, n)
    batch = a.shape[0]

    res = torch.zeros([batch, m, n]).float()

    row_arr = a.repeat([1, 1, n]).flatten()
    col_tmp = b.repeat([1, m, 1])
    col_tmp = col_tmp.reshape(-1, m, d, n)
    col_tmp = col_tmp.permute(0, 1, 3, 2)
    col_arr = col_tmp.flatten()

    prod = row_arr * col_arr
    del row_arr
    del col_arr
    prod = prod.reshape(-1, m, n, d)
    prod = (prod.view(torch.int32) & mask).view(torch.float32)
    res = prod.sum(-1)

    keep_shape.extend([m, n])
    res = res.reshape(keep_shape)
    return res.to(dtype)
