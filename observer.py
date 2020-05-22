import numpy as np
from numpy import linalg as LA


class SateliteObserver:

    def __init__(self, lat, long):
        # transform latitude and longitude to radians
        self.theta = long*np.pi/180
        self.phi = lat*np.pi/180

        self.transform_M = np.array([[np.sin(self.phi)*np.cos(self.theta), np.sin(self.phi)*np.sin(self.theta), -np.cos(self.phi)],
                                     [-np.sin(self.theta), np.cos(self.theta), 0],
                                     [np.cos(self.phi)*np.cos(self.theta), np.cos(self.phi)*np.sin(self.theta), np.sin(self.phi)]])
        self.T = np.array([[np.sin(self.phi)*np.cos(self.theta), np.sin(self.phi)*np.sin(self.theta), -np.cos(self.phi), 0, 0, 0],
                            [-np.sin(self.theta), np.cos(self.theta), 0, 0, 0, 0],
                            [np.cos(self.phi)*np.cos(self.theta), np.cos(self.phi)*np.sin(self.theta), np.sin(self.phi), 0, 0, 0],
                            [0, 0, 0, np.sin(self.phi) * np.cos(self.theta), np.sin(self.phi) * np.sin(self.theta), -np.cos(self.phi)],
                            [0, 0, 0, -np.sin(self.theta), np.cos(self.theta), 0],
                            [0, 0, 0, np.cos(self.phi) * np.cos(self.theta), np.cos(self.phi) * np.sin(self.theta),
                             np.sin(self.phi)]])
        self.R = 6371e3
        self.sat = [np.cos(self.phi)*np.cos(self.theta)*self.R, np.cos(self.phi)*np.sin(self.theta)*self.R, np.sin(self.phi)*self.R]

    def position_transform(self, r):
        return np.matmul(self.transform_M, r) - [0, 0, self.R]

    def h(self, x, error='on'):
        if len(x) != 3:
            r = [x[0], x[1], x[2]]
            error = 'off'
        else:
            r = x

        if error == 'on':
            sigma_d = np.random.normal(0, 400, size=1) # 50
            sigma_el = np.random.normal(0, 1e-2, size=1) # 1e-3
            sigma_az = np.random.normal(0, 1e-2, size=1) # 1e-3
        elif error == 'off':
            sigma_d = [0]
            sigma_el = [0]
            sigma_az = [0]

        y = [0, 0, 0]
        r_t = self.position_transform(r)
        a = LA.norm(r_t) + sigma_d
        y[0] = a[0]
        a = np.arcsin(r_t[2]/(y[0]-sigma_d[0])) + sigma_el
        y[1] = a[0]
        a = np.arctan(r_t[1]/(-r_t[0])) + sigma_az
        y[2] = a[0]
        return y

    def h_inv(self, d, el, az):
        rz = d*np.sin(el)
        rx = - d*np.cos(el)*np.cos(az)
        ry = d*np.cos(el)*np.sin(az)
        # rz = d * np.sin(el)
        # rx = ((d ** 2 - rz ** 2) / (1 + np.tan(az) ** 2)) ** 0.5
        # ry = -rx * np.tan(az)

        return [rx, ry, rz]


