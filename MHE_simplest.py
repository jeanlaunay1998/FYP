import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from system_dynamics import dynamics
from observer import SateliteObserver
from state_model import model


class MHE:
    def __init__(self, model, observer):
        self.alpha = 0.1  # random size of the step (to be changed by Hessian of the matrix)
        self.N = 10  # number of points in the horizon
        self.J = 0  # matrix to store cost function

        self.m = model  # copy direction of model to access all function of the class
        self.o = observer  # copy direction of observer to access all function of the class
        self.x_apriori = []
        self.x_init = []
        self.y = []
        self.a = []
        self.beta = []
        self.grad = []
        self.x_horizon = []
        self.x_solution = []

    def initialisation(self, y_measured, step):
        self.y = np.array(y_measured)[step-self.N:step, :]
        self.a = np.array(m.a)
        self.beta = m.beta

        self.x_apriori = np.array(m.Sk[step-self.N])
        self.x_init = self.x_apriori
        self.x_horizon = np.array(m.Sk)[step-self.N:step]

    def cost_function(self, x):
        if len(x) != len(self.x_apriori):
            a = np.zeros((2,3))
            for i in range(len(x)):
                a[int(i/3), i%3] = x[i]
            x = np.array(a)

        self.J = pow(LA.norm(x - self.x_apriori), 2) + pow(LA.norm(self.y[0] - self.o.h(x[0])), 2)
        x_iplus1 = x
        for i in range(self.N):
            x_iplus1[0], x_iplus1[1], self.a, self.beta = self.m.f(x_iplus1[0], x_iplus1[1], self.a, self.beta)
            self.J = self.J + pow(LA.norm(self.y[i] - self.o.h(x_iplus1[0])), 2)
        return self.J


    def density_constants(self, height):
        height = height - m.R
        if height < 9144:
            c1 = 1.227
            c2 = 1.093e-4
        else:
            c1 = 1.754
            c2 = 1.490e-4
        return [c1, c2]

# All functions below are used to compute the gradient of the cost function (an error exists in the derivation since the
    # real value is not working)
    def dacc_dr(self, r, v):
        norm_r = LA.norm(r)
        norm_v = LA.norm(v)

        # For legibility of the gradient constant terms across gradient are previously defined
        constant1 = m.G*m.M*pow(norm_r, -3)
        c1, c2 = self.density_constants(norm_r)
        constant2 = norm_v*c2*m.density_h(r)/(2*self.beta*norm_r)

        dA = constant1 * np.array([[-1 + pow(norm_r, -2)*r[0]*r[0], pow(norm_r, -2)*r[0]*r[1], pow(norm_r, -2)*r[0]*r[2]],
              [pow(norm_r, -2)*r[1]*r[0], -1 + pow(norm_r, -2)*r[1]*r[1], pow(norm_r, -2)*r[1]*r[2]],
              [pow(norm_r, -2)*r[2]*r[0], pow(norm_r, -2)*r[2]*r[1], -1 + pow(norm_r, -2)*r[2]*r[2]]])

        dB = np.array([[constant2[0]*v[0]*r[0], constant2[0]*v[0]*r[1], constant2[0]*v[0]*r[2]],
              [constant2[0]*v[1]*r[0], constant2[0]*v[1]*r[1], constant2[0]*v[1]*r[2]],
              [constant2[0]*v[2]*r[0], constant2[0]*v[2]*r[1], constant2[0]*v[2]*r[2]]])
        dadr = dA + dB
        return dadr

    def dacc_dv(self, r, v):
        norm_v = LA.norm(v)
        constant1 = -(m.density_h(r)/self.beta)
        dadv = np.array([[norm_v + pow(norm_v, -1)*v[0]*v[0], pow(norm_v, -1)*v[0]*v[1], pow(norm_v, -1)*v[0]*v[2]],
                        [pow(norm_v, -1)*v[1]*v[0], norm_v + pow(norm_v, -1)*v[1]*v[1], pow(norm_v, -1)*v[1]*v[2]],
                        [pow(norm_v, -1)*v[2]*v[0], pow(norm_v, -1)*v[2]*v[1], norm_v + pow(norm_v, -1)*v[2]*v[2]]])
        dadv = constant1*dadv
        return dadv

    def df(self, x):
        # x: point at which the derivative is evaluated
        r = x[0]
        v = x[1]

        # compute acceleration derivatives
        dadr = self.dacc_dr(r, v)
        dadv = self.dacc_dv(r, v)
        # total derivative
        dfdx = []
        for i in range(3):
            dfdx.append((1 + 0.5*pow(m.delta_t, 2)*dadr[i]).tolist() + (m.delta_t + 0.5*pow(m.delta_t, 2)*dadv[i]).tolist())
        for i in range(3):
            dfdx.append((m.delta_t*dadr[i]).tolist() + (1 + m.delta_t*dadr[i]).tolist())
        # for i in range(4): dfdx[i] = dfdx[i].tolist()
        return np.array(dfdx)

    def dh(self, x):
        # x: point at which the derivative is evaluated
        r = o.position_transform(x[0])

        norm_r = LA.norm(r)
        dhdx = [[r[0]/norm_r, r[1]/norm_r, r[2]/norm_r, 0, 0, 0]]
        constant1 = 1/(np.sqrt(1-pow(r[2]/norm_r, 2)))
        constant2 = -constant1*r[2]*pow(norm_r, -3)
        dhdx.append([constant2*r[0], constant2*r[1], constant1*(pow(norm_r, -1) + constant2*r[2]), 0, 0, 0])
        constant3 = 1/((1 + pow(r[1]/r[0], 2))*r[0])
        dhdx.append([constant3*r[1]/r[0], -constant3, 0, 0, 0, 0])
        dhdx = np.matmul(dhdx, o.T)
        return dhdx


    def gradient(self, x_o):
        a = 2*(x_o - self.x_apriori)
        self.grad = []
        for i in range(6): self.grad.append(a[int(i/3), i%3])
        self.grad = np.array(self.grad)

        dfdx_i = []
        dhdx_i = []
        h_i = []
        x_i = self.x_init
        beta_i = self.beta
        a_i = self.a

        for i in range(self.N):
            dfdx_i.append(self.df(x_i))
            dhdx_i.append(self.dh(x_i))
            h_i.append(o.h(x_i[0]))
            x_i[0], x_i[1], a_i, beta_i = self.m.f(x_i[0], x_i[1], a_i, beta_i)

        dhdx_i = np.array(dhdx_i)
        h_i = np.array(h_i)

        for i in range(self.N):
            dfdx_mult = dfdx_i[0]
            for j in range(i):
                dfdx_mult = np.matmul(dfdx_mult, dfdx_i[j+1])
            A = np.matmul(np.transpose(dfdx_mult), np.transpose(dhdx_i[i]))
            B = np.transpose(self.y[i] - h_i[i])
            C = 2*np.matmul(A, B)
            self.grad = self.grad + C

    def step_optimization(self, y, e, step):
        self.initialisation(y, step)
        for i in range(100):
            print(self.cost_function(self.x_init))
            self.gradient(self.x_init)
            self.x_init[0] = self.x_init[0] - self.alpha*self.grad[0:3]
            self.x_init[1] = self.x_init[1] - self.alpha*self.grad[3:6]



# main
height = 80e3
d = dynamics(height, 22, 0, 6000, -5, 60)
o = SateliteObserver(40.24, 3.42, d)
m = model(d.r, d.v, d.beta)
opt = MHE(m, o)


step = 1
t_lim = 100
t = 0
y_real = [o.h(d.r)]
y_model = [o.h(m.r)]
y_mhe = y_model


while height > 5000 and t < t_lim:
    step = step + 1
    d.step_update(d.v, d.r)
    m.step_update()

    y_real.append(o.h(d.r))
    y_model.append(o.h(m.r))
    y_mhe.append(o.h(m.r))

    height = d.h[len(d.h)-1]
    t = t + d.delta_t
    if step>100:
        opt.initialisation(y_real, step)
        x_initial = []
        for i in range(6): x_initial.append(opt.x_init[int(i / 3), i % 3])
        x_initial = np.array(x_initial)
        res = minimize(opt.cost_function, x_initial, method='nelder-mead', options = {'xatol': 1e-4, 'adaptive':True})
        # opt.step_optimization(y_real, e, step)


time = np.linspace(0, t, len(d.h))

plt.figure(1)
plt.plot(time, d.beta, 'b', label='Ballistic coef')
plt.plot(time, m.ballistic, 'r', label='Ballistic coef')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.plot(time, d.h, 'b', label='Height real (m)')
plt.plot(time, m.h, 'r', label='Height model (m)')
plt.legend(loc='best')
plt.show()

fig, ax = plt.subplots(3)
yplot = np.array(y_real)
ax[0].plot(time, yplot[:, 0], 'b', label='Real distance')
ax[1].plot(time, yplot[:, 1], 'b', label='Real elevation angle')
ax[2].plot(time, yplot[:, 2], 'b', label='Real azimuth angle')

yplot = np.array(y_model)
ax[0].plot(time, yplot[:, 0], 'r', label='Estimated distance')
ax[1].plot(time, yplot[:, 1], 'r', label='Estimated elevation angle')
ax[2].plot(time, yplot[:, 2], 'r', label='Estimated azimuth angle')
plt.legend(loc='best')
plt.show()

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