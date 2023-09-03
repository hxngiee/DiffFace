import numpy as np
import torch
import torch.nn as nn


def softargmax2d(input, beta=100, dtype=torch.float32):
    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)))
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)))

    device = input.get_device()
    if device >= 0:
        indices_r = indices_r.to(device)
        indices_c = indices_c.to(device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result.type(dtype)


def softargmax1d(input, beta=100):
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result

