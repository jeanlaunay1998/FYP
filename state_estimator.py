import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from system_dynamics import dynamics
from observer import SateliteObserver


class estimator:
    def __init__(self, r_o, v_o, beta_o):
        # r_o initial estimated position
        # v_o initial estimated velocity
        # beta_o initial estimated ballistic coefficient

        # Useful constants
        self.I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.G = 6.673e-11
        self.M = 5.972e24
        self.R = 6371e3

        self.delta_t = 0.01
        self.beta_o = beta_o[0]

        self.dbeta = np.random.normal(0, pow(0.01 * self.beta_o, 2), size=1)
        self.dr = np.random.normal(0, pow(self.delta_t, 3)/3 + 0.5*pow(self.delta_t, 2), size=1)
        self.dv = np.random.normal(0, self.delta_t + 0.5*pow(self.delta_t, 2), size=1)

        self.r = r_o + self.dr
        self.r = self.r.tolist()
        self.v = v_o + self.dv
        self.v = self.v.tolist()

        self.beta = beta_o + self.dbeta
        self.a = self.acceleration(self.v, self.r, self.beta)
        self.Sk = [self.r, self.v]
        self.h = [LA.norm(self.r)-self.R]
        self.ballistic = [self.beta[0]]

    def density_h(self,r):
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

    def acceleration(self, v, r, beta):
        a = -np.multiply((self.G*self.M)/pow(LA.norm(r), 3), r)
        b = - np.multiply(self.density_h(r) * LA.norm(v)/(2*beta[0]), v)
        acc =  a + b
        return acc.tolist()

    def step_update(self):

        self.dbeta = 0 # np.random.normal(0, pow(0.005 * self.beta_o, 2), size=1)
        self.dr = np.random.normal(0, pow(self.delta_t, 3) / 3 + 0.5 * pow(self.delta_t, 2), size=1)
        self.dv = np.random.normal(0, self.delta_t + 0.5 * pow(self.delta_t, 2), size=1)

        self.r = np.multiply(self.I, self.r) + self.delta_t*np.multiply(self.I, self.v) + pow(self.delta_t, 2)/2 * np.multiply(self.I, self.a) + self.dr
        self.r = self.r.tolist()
        self.v = np.multiply(self.I, self.v) + self.delta_t*np.multiply(self.I, self.a) + self.dv
        self.v.tolist()


        self.beta = self.beta + self.dbeta
        self.Sk.append([self.r, self.v])

        self.a = self.acceleration(self.v, self.r, self.beta)

        # Variables used for plots
        self.h.append(LA.norm(self.r) - self.R)
        self.ballistic.append(self.beta[0])




# main
height = 80e3
d = dynamics(height, 40.24, 3.42, 6000, -5, 60)
o = SateliteObserver(40.24, 3.42, d)
e = estimator(d.r, d.v, d.beta)
t_lim = 1000
t = 0
y_real = o.h(d.r)
y_estimate = o.h(e.r)

while height>5000 and t<t_lim:
    d.step_update(d.v, d.r)
    e.step_update()

    y_real.append(o.h(d.r))
    y_estimate.append(o.h(e.r))

    height = d.h[len(d.h)-1]
    t = t + d.delta_t


# time = np.linspace(0,t,len(d.h))
# plt.figure(1)
# y_plot = np.array(o.y)
# print(y_plot[0][0])
# plt.plot(time, y_plot[:][0], 'b', label='Ballistic coef')
# plt.legend(loc='best')
# plt.show()

time = np.linspace(0,t,len(d.h))
plt.figure(1)
plt.plot(time, d.beta, 'b', label='Ballistic coef')
plt.plot(time, e.ballistic, 'r', label='Ballistic coef')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.plot(time, d.h, 'b', label='Height real (m)')
plt.plot(time, e.h, 'r', label='Height estimator (m)')
plt.legend(loc='best')
plt.show()

# plotX = np.array(d.x)
from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(3)
# ax = fig.add_subplot(111, projection='3d')
# # ax.plot(plotX[:, 2, 0], plotX[:, 2, 1], plotX[:, 2, 2])
#
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = np.cos(u)*np.sin(v)*d.R
# y = np.sin(u)*np.sin(v)*d.R
# z = np.cos(v)*d.R
# ax.plot_wireframe(x, y, z, color="r")

# X = np.array([[o.sat[0]], [d.r[0]]])
# Y = np.array([[o.sat[1]], [d.r[1]]])
# Z = np.array([[o.sat[2]], [d.r[2]]])
# ax.plot_wireframe(X, Y, Z, color="g")

# X = np.array([[0], [d.r[0]]])
# Y = np.array([[0], [d.r[1]]])
# Z = np.array([[0], [d.r[2]]])
# ax.plot_wireframe(X, Y, Z, color="g")
#
# X = np.array([[o.sat[0]], [0]])
# Y = np.array([[o.sat[1]], [0]])
# Z = np.array([[o.sat[2]], [0]])
# ax.plot_wireframe(X, Y, Z, color="b")
# plt.show()