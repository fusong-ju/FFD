import math
import torch
from torch import nn


class VonMises1D:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x):
        z = math.log(1.0 / (math.sqrt(2 * math.pi) * self.sigma))
        return z + (torch.cos(x - self.mu) - 1) / (self.sigma ** 2)
