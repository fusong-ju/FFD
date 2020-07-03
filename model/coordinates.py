import logging

import torch
from torch import nn


class Coordinates:
    def __init__(self, N, CA, C, CB):
        """
        shape: (B, L, 3)
        """
        self.C = C
        self.N = N
        self.CA = CA
        self.CB = CB


class ResiduePose(nn.Module):
    def __init__(self, translation, quaternion=None):
        """
        shape: (B, L, 3)
        """
        super().__init__()
        B, L = translation.shape[:2]
        self.translation = nn.Parameter(translation, requires_grad=True)
        self.quaternion = nn.Parameter(quaternion, requires_grad=True)

    def normalize(self):
        with torch.no_grad():
            center = torch.mean(self.translation, dim=1, keepdims=True)
            self.translation -= center
            norm = torch.norm(self.quaternion, 2, dim=-1).unsqueeze(-1)
            self.quaternion /= norm + 1e-6

    def to_coord(self):
        self.normalize()
        device = self.translation.device
        R = torch.zeros(*self.quaternion.shape[:-1], 3, 3, device=device)
        r = self.quaternion[..., 0]
        i = self.quaternion[..., 1]
        j = self.quaternion[..., 2]
        k = self.quaternion[..., 3]
        R[..., 0, 0] = 1 - 2 * (j**2 + k**2)
        R[..., 0, 1] = 2 * (i * j - k * r)
        R[..., 0, 2] = 2 * (i * k + j * r)
        R[..., 1, 0] = 2 * (i * j + k * r)
        R[..., 1, 1] = 1 - 2 * (i**2 + k**2)
        R[..., 1, 2] = 2 * (j * k - i * r)
        R[..., 2, 0] = 2 * (i * k - j * r)
        R[..., 2, 1] = 2 * (j * k + i * r)
        R[..., 2, 2] = 1 - 2 * (i**2 + j**2)
        internal_coord = {
            "N": [1.460091, 0.0, 0.0],
            "CA": [0.0, 0.0, 0.0],
            "C": [-0.56431316, 1.41695817, 0.0],
            "CB": [-0.52426314, -0.76611338, 1.20561194],
        }
        d = type('coordinate', (), {})()
        for key in ["N", "CA", "C", "CB"]:
            pos = torch.tensor(internal_coord[key], device=device)
            d.__dict__[key] = (torch.matmul(R, pos) + self.translation)
        return d


if __name__ == "__main__":
    import numpy as np
    a = torch.zeros(1, 1, 3)
    b = torch.zeros(1, 1, 4)
    r = ResiduePose(a, b)
    coord = r.to_coord()
    print(coord)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    # print("N=", repr(r.N.detach().numpy()[0][0])[6:-1])
    # print("CA=", repr(r.CA.detach().numpy()[0][0])[6:-1])
    # print("C=", repr(r.C.detach().numpy()[0][0])[6:-1])
    # print("CB=", repr(r.CB.detach().numpy()[0][0])[6:-1])
