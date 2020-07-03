import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.geometric import pairwise_distance
from . import spline
from common import config
from common.config import EPS


class DistanceRestraint(nn.Module):
    """ Cb-Cb distance """

    def __init__(self, pred_dist):
        """
        pred_dist has shape (L, L, 37)
        """
        super().__init__()
        _bins = np.linspace(2.25, 19.75, num=36)
        self.cutoffs = torch.tensor([0] + _bins.tolist()).float()

        _x = self.cutoffs.numpy().tolist()
        _ref = pred_dist[:, :, -2:-1] * np.array((_bins / _bins[-1]) ** 1.57)[None, None]
        _y = -np.log(pred_dist[:, :, :-1] + EPS) + np.log(_ref + EPS)
        _y = np.concatenate([_y[:, :, :1] - np.log(EPS), _y], axis=-1)
        self.coeff = nn.Parameter(spline.cubic_spline(_x, _y), requires_grad=False)

        L = pred_dist.shape[0]
        _filter = torch.tensor(pred_dist[:, :, -1] < config.DIST_CUT)
        self.mask = nn.Parameter(
            torch.triu(torch.ones((L, L)).bool(), diagonal=1).__and__(_filter),
            requires_grad=False,
        )

        self.cutoffs = nn.Parameter(self.cutoffs, requires_grad=False)

    def __str__(self):
        return "CBCB constraints: %i" % torch.sum(self.mask).item()

    def forward(self, coord):
        B = coord.CB.shape[0]
        x_idx, y_idx = torch.where(self.mask)
        x_CB, y_CB = coord.CB[:, x_idx].view(-1, 3), coord.CB[:, y_idx].view(-1, 3)
        x_idx, y_idx = x_idx.repeat(B), y_idx.repeat(B)
        d_CB = torch.norm(x_CB - y_CB, dim=-1)
        mask = d_CB <= self.cutoffs[-1]
        x_idx, y_idx, d_CB = x_idx[mask], y_idx[mask], d_CB[mask]
        coeff = self.coeff[x_idx, y_idx]
        cbcb_potential = torch.sum(spline.evaluate(coeff, self.cutoffs, d_CB))

        return {
            "cbcb_distance": cbcb_potential,
        }
