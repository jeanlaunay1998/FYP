from system_dynamics import dynamics
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

        self.y = self.h(dynamics.r)

    def position_transform(self, r):
        return np.matmul(self.transform_M, r) - [0, 0, self.R]

    def h(self, r):
        y = [0, 0, 0]
        r_t = self.position_transform(r)
        y[0] = LA.norm(r_t)
        y[1] = np.arcsin(r_t[2]/y[0])
        y[2] = np.arctan(r_t[1]/(-r[0]))
        return y

# main
height = 80e3
d = dynamics(height, 40.24, 3.42, 6000, -5, 60)
o = SateliteObserver(40.24, 3.42, d)

t_lim = 1000
t = 0

while height>5000 and t<t_lim:
    d.delta_o = np.random.normal(0, pow(0.01 * d.beta_o, 2), size=1)
    d.a_res = 0 # np.random.normal(0, pow(0.01 * LA.norm(d.a), 2), size=1)
    d.step_update(d.v, d.r)

    height = d.h[len(d.h)-1]
    o.y.append(o.h(d.r))
    t = t + d.delta_t


# import matplotlib.pyplot as plt
# time = np.linspace(0,t,len(d.h))
# plt.figure(1)
y_plot = np.array(o.y)
print(y_plot[0][0])
# plt.plot(time, y_plot[:][0], 'b', label='Ballistic coef')
# plt.legend(loc='best')
# plt.show()

# time = np.linspace(0,t,len(d.h))
# plt.figure(1)
# plt.plot(time, d.beta, 'b', label='Ballistic coef')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(2)
# plt.plot(time, d.h, 'b', label='Height (m)')
# # plt.plot(time,sol[:,1],'r',label='Omega(t)')
# plt.legend(loc='best')
# plt.show()
#
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
