import numpy as np


class Prediction:
    """ Prediction of upstream neural network, such as CopularNet"""

    def __init__(self, geo_npz=None):
        self.geo = geo_npz

    @property
    def cbcb(self):
        return self.geo["cbcb"]

    @property
    def omega(self):
        return self.geo["omega"]

    @property
    def theta(self):
        return self.geo["theta"]

    @property
    def phi(self):
        return self.geo["phi"]
