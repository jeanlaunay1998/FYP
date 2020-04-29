import numpy as np
from numpy import linalg as LA
import sys

class model:
    def __init__(self, r_o, v_o, beta_o, delta):
        # r_o initial position from model
        # v_o initial velocity from theoretical model
        # beta_o initial estimated ballistic coefficient

        # Useful constants
        self.I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.G = 6.673e-11
        self.M = 5.972e24
        self.R = 6371e3

        self.delta_t = delta

        self.r = r_o
        self.v = v_o

        self.beta = beta_o
        self.a = []
        self.Sk = []
        self.h = []
        self.ballistic = []

    def reinitialise(self, y_minus1, y_o, o, measurement_lapse):
        distance = y_o[0]
        elevation = y_o[1]
        azimuth = y_o[2]
        rz = distance * np.sin(elevation)
        rx = ((distance ** 2 - rz ** 2) / (1 + np.tan(azimuth) ** 2)) ** 0.5
        ry = -rx * np.tan(azimuth)
        r = np.matmul(np.linalg.inv(o.transform_M), [rx, ry, rz + o.R])

        distance = y_minus1[0]
        elevation = y_minus1[1]
        azimuth = y_minus1[2]
        rz = distance * np.sin(elevation)
        rx = ((distance ** 2 - rz ** 2) / (1 + np.tan(azimuth) ** 2)) ** 0.5
        ry = -rx * np.tan(azimuth)
        rminus = np.matmul(np.linalg.inv(o.transform_M), [rx, ry, rz + o.R])

        self.r = r
        self.v = (r - rminus)/measurement_lapse
        self.a = self.acceleration(self.r, self.v, self.beta)
        self.Sk = [[self.r.tolist(), self.v.tolist()]]
        self.h = [LA.norm(self.r) - self.R]
        self.ballistic = [self.beta]

    def density_h(self, r):
        height = LA.norm(r) - self.R

        if height<9144:
            c1 = 1.227
            c2 = 1.093e-4
            rho = c1*np.exp(-c2*height)
        else:
            c1 = 1.754
            c2 = 1.490e-4
            rho = c1*np.exp(-c2*height)
        return rho

    def acceleration(self, r, v, beta):
        a = -np.multiply((self.G*self.M)/pow(LA.norm(r), 3), r)
        b = - np.multiply(self.density_h(r)*LA.norm(v)/(2*beta), v)
        acc =  a + b
        return acc.tolist()

    def f(self, r, v, beta, error='on', w=0):
        a = self.acceleration(r, v, beta)
        if error == 'on':
            dr = np.random.normal(0, pow(self.delta_t, 3) / 3 + 0.5 * pow(self.delta_t, 2), size=1)
            dv = np.random.normal(0, self.delta_t + 0.5 * pow(self.delta_t, 2), size=1)
        elif error== 'off':
            dr = 0
            dv = 0
        else:
            print('Error: wrong argument. Use off to compute step update without random error. ')
            sys.exit()

        r_return = r + np.multiply(self.delta_t, v) + np.multiply(pow(self.delta_t, 2) / 2, a) + dr
        r_return = r_return.tolist()
        v_return = v + np.multiply(self.delta_t, a) + dv
        v_return = v_return.tolist()

        beta_return = beta + w #+ self.dbeta
        a_return = self.acceleration(r_return, v_return, beta_return)

        return [r_return, v_return, a_return, beta_return]


    def step_update(self, error='on'):
        self.r, self.v, self.a, self.beta = self.f(self.r, self.v, self.beta, error)
        self.Sk.append([self.r, self.v])

        # Variables used for plots
        self.h.append(LA.norm(self.r) - self.R)
        self.ballistic.append(self.beta)
