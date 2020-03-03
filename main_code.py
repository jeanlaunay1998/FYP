import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

from system_dynamics import dynamics
from observer import SateliteObserver
from state_model import model
from MHE_simplest import MHE

height = 80e3
d = dynamics(height, 22, 0, 6000, -5, 60)
o = SateliteObserver(40.24, 3.42, d)
m = model(d.r, d.v, d.beta)
opt = MHE(m, o)


t_lim = 200
t = 0.00

y_real = [o.h(d.r)]
y_model = [o.h(m.r)]
y_mhe = []
step = int(1)
delta = int(0)

time_est = []
time = [0]


while height > 5000 and t < t_lim:
    d.step_update(d.v, d.r)
    m.step_update()#
    delta = delta + 1

    # update stopping criteria
    t = t + d.delta_t
    height = d.h[len(d.h)-1]
    print(t)

    # measurements are only taken every 0.1 seconds
    if delta == 100:

        delta = int(0)
        step = step + 1
        y_real.append(o.h(d.r))
        y_model.append(o.h(m.r))
        time.append(t)

        if step >= opt.N:
            opt.initialisation(y_real, step)
            x_initial = []
            for i in range(6): x_initial.append(opt.x_init[int(i / 3), i % 3])

            x_initial = np.array(x_initial)
            res = minimize(opt.cost_function, opt.x_init, method='nelder-mead', options = {'xatol': 1e-2, 'adaptive' : True})

            for j in range(6): opt.x_solution[int(j/3), j % 3] = res.x[j]
            time_est.append(t)
            y_mhe.append(o.h(opt.x_solution[0]))



# time = np.linspace(0, t, len(d.h))
#
# plt.figure(1)
# plt.plot(time, d.beta, 'b', label='Ballistic coef')
# plt.plot(time, m.ballistic, 'r', label='Ballistic coef')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(2)
# plt.plot(time, d.h, 'b', label='Height real (m)')
# plt.plot(time, m.h, 'r', label='Height model (m)')
# plt.legend(loc='best')
# plt.show()

fig, ax = plt.subplots(3)
yplot = np.array(y_real)
time = np.linspace(0, t, len(yplot))
ax[0].plot(time, yplot[:, 0], 'b', label='Real distance')
ax[1].plot(time, yplot[:, 1], 'b', label='Real elevation angle')
ax[2].plot(time, yplot[:, 2], 'b', label='Real azimuth angle')

yplot = np.array(y_model)
ax[0].plot(time, yplot[:, 0], 'r', label='Model distance')
ax[1].plot(time, yplot[:, 1], 'r', label='Model elevation angle')
ax[2].plot(time, yplot[:, 2], 'r', label='Model azimuth angle')
plt.legend(loc='best')

yplot = np.array(y_mhe)
time = np.linspace(0, t-opt.N*d.delta_t, len(yplot))
ax[0].plot(time_est, yplot[:, 0], 'g', label='Estimated distance')
ax[1].plot(time_est, yplot[:, 1], 'g', label='Estimated elevation angle')
ax[2].plot(time_est, yplot[:, 2], 'g', label='Estimated azimuth angle')
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