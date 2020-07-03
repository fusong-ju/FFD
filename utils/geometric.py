import torch


def distance(x, y):
    """
    in_shape: (B, *, 3)
    out_shape: (B, *)
    """
    return torch.norm(x - y, dim=-1)


def pairwise_distance(x, y=None):
    """
    in_shape: (B, L, k)
    out_shape: (B, L, L)
    """
    if y is None:
        y = x
    return torch.norm(x[:, :, None, :] - y[:, None, :, :], dim=-1)


def calc_angle(v1, v2, v3, x_idx=None, y_idx=None, eps=1e-6):
    """
    Calculate the angle between 3 vectors.
    v1, v2, v3: shape (..., 3)
    x_idx, y_idx: shape (...), additional information of vectors.
    return: (x_idx, y_idy, angle)
    """
    x = v1 - v2
    y = v3 - v2

    mask = (torch.norm(x, dim=-1) > eps).__and__(torch.norm(y, dim=-1) > eps)

    x, y = x[mask], y[mask]

    if x_idx is not None:
        x_idx, y_idx = x_idx[mask], y_idx[mask]
    nx = torch.norm(x, dim=-1)
    ny = torch.norm(y, dim=-1)
    c = torch.sum(x * y, dim=-1) / (nx * ny)

    good_grad = 1 - c * c > eps
    if x_idx is not None:
        x_idx, y_idx = x_idx[good_grad], y_idx[good_grad]
    return (x_idx, y_idx, torch.acos(c[good_grad]))


def calc_dihedral(v1, v2, v3, v4, x_idx=None, y_idx=None, eps=1e-6):
    """
    Calculate the dihedral angle between 4 vectors.
    v1, v2, v3, v4: shape (..., 3)
    x_idx, y_idx: shape (...), additional information of vectors.
    return: (x_idx, y_idy, dihedral)
    """
    x = v2 - v1
    y = v3 - v2
    z = v4 - v3

    mask = torch.norm(x, dim=-1) > eps
    mask = mask.__and__(torch.norm(y, dim=-1) > eps)
    mask = mask.__and__(torch.norm(z, dim=-1) > eps)

    x, y, z = x[mask], y[mask], z[mask]

    if x_idx is not None:
        x_idx, y_idx = x_idx[mask], y_idx[mask]

    c_xy = torch.cross(x, y)
    c_yz = torch.cross(y, z)
    sin = torch.sum(y * torch.cross(c_xy, c_yz), dim=-1)
    cos = torch.sum(c_xy * c_yz, dim=-1) * torch.norm(y, dim=-1)

    good_grad = sin * sin + cos * cos > eps
    if x_idx is not None:
        x_idx, y_idx = x_idx[good_grad], y_idx[good_grad]
    return (x_idx, y_idx, torch.atan2(sin[good_grad], cos[good_grad]))
