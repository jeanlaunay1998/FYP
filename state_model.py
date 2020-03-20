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
        self.beta_o = beta_o

        self.dbeta = np.random.normal(0, pow(0.01 * self.beta_o, 2), size=1)
        self.dr = np.random.normal(0, pow(self.delta_t, 3)/3 + 0.5*pow(self.delta_t, 2), size=1)
        self.dv = np.random.normal(0, self.delta_t + 0.5*pow(self.delta_t, 2), size=1)

        self.r = r_o + self.dr
        self.r = self.r.tolist()
        self.v = v_o + self.dv
        self.v = self.v.tolist()

        self.beta = beta_o + self.dbeta
        self.a = self.acceleration(self.r, self.v, self.beta)
        self.Sk = [[self.r, self.v]]
        self.h = [LA.norm(self.r)-self.R]
        self.ballistic = [self.beta[0]]

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
        b = - np.multiply(self.density_h(r)*LA.norm(v)/(2*beta[0]), v)
        acc =  a + b
        return acc.tolist()

    def f(self, r, v, a, beta, error='on'):
        if error == 'on':
            self.dbeta = 0  # np.random.normal(0, pow(0.005 * self.beta_o, 2), size=1)
            self.dr = np.random.normal(0, pow(self.delta_t, 3) / 3 + 0.5 * pow(self.delta_t, 2), size=1)
            self.dv = np.random.normal(0, self.delta_t + 0.5 * pow(self.delta_t, 2), size=1)
        elif error== 'off':
            self.dbeta = 0
            self.dr = 0
            self.dv = 0

        r_return = r + np.multiply(self.delta_t, v) + np.multiply(pow(self.delta_t, 2) / 2, a) + self.dr
        r_return = r_return.tolist()
        v_return = self.v + np.multiply(self.delta_t, a) + self.dv
        v_return = v_return.tolist()

        beta_return = beta #+ self.dbeta
        a_return = self.acceleration(r, v, beta)

        return [r_return, v_return, a_return, beta_return]


    def step_update(self):
        self.r, self.v, self.a, self.beta = self.f(self.r, self.v, self.a, self.beta)
        self.Sk.append([self.r, self.v])

        # Variables used for plots
        self.h.append(LA.norm(self.r) - self.R)
        self.ballistic.append(self.beta[0])