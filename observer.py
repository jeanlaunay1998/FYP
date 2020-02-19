import numpy as np
from numpy import linalg as LA


class SateliteObserver:

    def __init__(self, lat, long, dynamics):
        # transform latitude and longitude to radians
        self.theta = long*np.pi/180
        self.phi = lat*np.pi/180

        self.transform_M = np.array([[np.sin(self.phi)*np.cos(self.theta), np.sin(self.phi)*np.sin(self.theta), -np.cos(self.phi)],
                                     [-np.sin(self.theta), np.cos(self.theta), 0],
                                     [np.cos(self.phi)*np.cos(self.theta), np.cos(self.phi)*np.sin(self.theta), np.sin(self.phi)]])
        self.R = 6371e3
        self.sat = [np.cos(self.phi)*np.cos(self.theta)*self.R, np.cos(self.phi)*np.sin(self.theta)*self.R, np.sin(self.phi)*self.R]

    def position_transform(self, r):
        return np.matmul(self.transform_M, r) - [0, 0, self.R]

    def h(self, r):
        y = [0, 0, 0]
        r_t = self.position_transform(r)
        y[0] = LA.norm(r_t)
        y[1] = np.arcsin(r_t[2]/y[0])
        y[2] = np.arctan(r_t[1]/(-r_t[0]))
        return y


