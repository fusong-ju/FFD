import math
import torch
from torch import nn
import torch.nn.functional as F
from common import constants
from utils.geometric import distance, calc_angle, calc_dihedral
from .von_mises import VonMises1D


class BondRestraint(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def gaussian_loss(self, x, mean):
        m = torch.distributions.normal.Normal(mean, self.sigma)
        return -torch.sum(m.log_prob(x))

    def circular_gaussian_loss(self, x, mean):
        m = VonMises1D(mean, self.sigma)
        return -torch.sum(m.log_prob(x))

    def bond_length_restraint(self, coord):
        N, C = coord.N, coord.C
        C_N = self.gaussian_loss(distance(C[:, :-1], N[:, 1:]), constants.L_C_N)
        return C_N

    def bond_angle_restraint(self, coord):
        N, CA, C = coord.N, coord.CA, coord.C
        CA_C_N = self.gaussian_loss(
            calc_angle(CA[:, :-1], C[:, :-1], N[:, 1:])[-1], constants.A_CA_C_N
        )
        C_N_CA = self.gaussian_loss(
            calc_angle(C[:, :-1], N[:, 1:], CA[:, 1:])[-1], constants.A_C_N_CA
        )
        omega = self.circular_gaussian_loss(
            calc_dihedral(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:])[-1],
            constants.OMEGA,
        )
        return CA_C_N + C_N_CA + omega

    def forward(self, coord):
        return {
            "bond_length": self.bond_length_restraint(coord),
            "bond_angle": self.bond_angle_restraint(coord),
        }
