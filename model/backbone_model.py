from torch import nn
from common import config
from collections import defaultdict
from .bond_restraint import BondRestraint
from .distance_restraint import DistanceRestraint
from .orientation_restraint import OmegaRestraint, ThetaRestraint, PhiRestraint


class BackBoneModel(nn.Module):
    def __init__(self, pred_dist=None, pred_omega=None, pred_theta=None, pred_phi=None, bond_sigma=0.1):
        super(BackBoneModel, self).__init__()
        self.bond_r = BondRestraint(sigma=bond_sigma)
        self.dist_r = DistanceRestraint(pred_dist)
        self.omega_r = OmegaRestraint(pred_omega)
        self.theta_r = ThetaRestraint(pred_theta)
        self.phi_r = PhiRestraint(pred_phi)
        self.weights = defaultdict(lambda: 1)

    def forward(self, pose, use_distance=True, use_orientation=True):
        coord = pose.to_coord()
        losses = {}
        if use_distance:
            losses.update(self.dist_r(coord))
        if use_orientation:
            losses.update(self.omega_r(coord))
            losses.update(self.theta_r(coord))
            losses.update(self.phi_r(coord))
        losses.update(self.bond_r(coord))
        for k, v in losses.items():
            losses[k] = v * self.weights[k]
        return losses

    def set_bond_sigma(self, sigma):
        self.bond_r.sigma = sigma

    def set_weight(self, key, value):
        self.weights[key] = value

    def clear_weight(self):
        self.weights = defaultdict(lambda: 1.0)

    def set_default_weight(self):
        self.clear_weight()
        self.set_weight("cbcb_distance", 5.0)
        self.set_weight("pairwise_omega", 1.0)
        self.set_weight("pairwise_theta", 1.0)
        self.set_weight("pairwise_phi", 1.0)

    def __str__(self):
        strs = []
        for x in [self.dist_r, self.omega_r, self.theta_r, self.phi_r]:
            if x is not None:
                strs += [str(x)]
        return ", ".join(strs)
