import scipy.interpolate
import torch
from torch import nn
import torch.nn.functional as F


def cubic_spline(x, y, period=False):
    """Calculates cubic spline
    x: shape (C, ), breakpoints.
    y: shape (..., C - 1), values.

    return: shape (..., C - 1, 4), coefficients of the polynomials.
    """

    c = scipy.interpolate.CubicSpline(x, y, axis=-1, bc_type="not-a-knot" if not period else "periodic").c
    c = torch.tensor(c).float().permute(2, 3, 1, 0)
    return c


def evaluate(c, b, x):
    """
    c: shape (..., C - 1, 4), coefficients of the polynomials.
    b: shape(C, ), breakpoints.
    x: shape (...), points to evaluate.

    return: shape(...), interpolated values.
    """
    idx = torch.searchsorted(b[None].contiguous(), x[None].detach().contiguous())[0] - 1
    idx = torch.clamp(idx, 0, c.shape[-2] - 1)

    x = x - b[idx]
    onehot = torch.eye(c.shape[-2], device=x.device, dtype=bool)
    c = c[onehot[idx]]

    ret = c[:, 3] + c[:, 2] * x
    t_x = x * x
    ret += c[:, 1] * t_x
    t_x = t_x * x
    ret += c[:, 0] * t_x
    return ret
