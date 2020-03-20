import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize
import numpy as np
from numpy import linalg as LA
import sys

from system_dynamics import dynamics
from observer import SateliteObserver
from state_model import model
from MHE_simplest import MHE

t_lim = 200

t = 0.00
step = int(1)  # number of measurements so saf
delta = int(0)
measurement_lapse = 0.5  # time lapse between every measurement

height = 80e3
d = dynamics(height, 22, 0, 6000, -5, 60)
o = SateliteObserver(40.24, 3.42, d)

# initialise model with errors
p = 1 - (70000 + d.R)/LA.norm(d.r)
initialr = [0,0,0]
initialr[0] = d.r[0] + np.random.normal(0, p*d.r[0], size=1)[0]
initialr[1] = d.r[1] + np.random.normal(0, p*d.r[1], size=1)[0]
initialr[2] = d.r[2] + np.random.normal(0, p*d.r[2], size=1)[0]

initialv = [0, 0, 0]
initialv[0] = d.v[0] + np.random.normal(0, 0.15*LA.norm(d.v[0]), size=1)[0]
initialv[1] = d.v[1] + np.random.normal(0, 0.15*LA.norm(d.v[1]), size=1)[0]
initialv[2] = d.v[2] + np.random.normal(0, 0.15*LA.norm(d.v[2]), size=1)[0]

initialbeta = d.beta[0] + np.random.normal(0, 0.1*d.beta[0], size=1)[0]

m = model(initialr, initialv, initialbeta, measurement_lapse)
opt = MHE(m, o, measurement_lapse)


time = [0]
y_real = [o.h(d.r)]
y_model = [o.h(m.r)]

y_mhe = []
state_estimate = []
time_est = []

true_cost = []
model_cost = []
estimate_cost = []

real_x = [[d.r, d.v]]
while height > 5000 and t < t_lim:
    d.step_update(d.v, d.r)
    delta = delta + 1

    # update stopping criteria
    t = t + d.delta_t
    height = d.h[len(d.h)-1]

    # measurements are only taken every 0.5 seconds (in the interest of time)
    if delta == measurement_lapse/d.delta_t:
        print(t-t_lim)

        delta = int(0)
        m.step_update() # the model is updated every 0.5 seconds (problem with discretization)

        step = step + 1
        y_real.append(o.h(d.r))
        y_model.append(o.h(m.r))

        if y_model[len(y_model)-1][0]> 1e10:
            print('Error in the model')
            sys.exit()

        time.append(t)
        real_x.append([d.r, d.v])

        if step >= opt.N+1: # MHE is entered only when there exists sufficient measurements over the horizon

            # print(t_lim-t)
            opt.initialisation(y_real, step)
            x_initial = []

            # --------------------------------------- #
            # a = opt.cost_function(opt.x_apriori)
            # print('gradient')
            # print(opt.gradient([d.r, d.v]))
            # print('aaaaaa')
            # eps = 0.001
            # plus_eps = [[d.r[0], d.r[1], d.r[2]], [d.v[0]+eps, d.v[1], d.v[2]]]
            # minus_eps = [[d.r[0], d.r[1], d.r[2]], [d.v[0]-eps, d.v[1], d.v[2]]]
            # A = opt.cost_function(plus_eps)
            # B = opt.cost_function(minus_eps)
            # derivative1 = (A - B) / (2*eps)
            # print(derivative1)
            # sys.exit()
            # --------------------------------------- #

            # optimisation
            for i in range(6): x_initial.append(opt.x_init[int(i / 3), i % 3])
            x_initial = np.array(x_initial)

            res = minimize(opt.cost_function, opt.x_init, method='BFGS', jac=opt.gradient, options = {'gtol': 1e-6})
            for j in range(6): opt.x_solution[int(j/3), j % 3] = res.x[j]

            # store points to analyse later
            state_estimate.append(res.x)
            time_est.append(t-opt.N*d.delta_t)
            y_mhe.append(o.h(opt.x_solution[0]))
            estimate_cost.append(opt.cost_function(opt.x_solution))
            true_cost.append(opt.cost_function(real_x[len(real_x)-1-opt.N]))
            model_cost.append(opt.cost_function(np.array(m.Sk[len(m.Sk)-1-opt.N*opt.inter_steps])))

# time = np.linspace(0, t, len(d.h))

# plt.figure(1)
# time = np.linspace(0, t, len(d.beta))
# plt.plot(time, d.beta, 'g', label='Ballistic coef')
# # plt.plot(time, m.ballistic, 'r', label='Ballistic coef')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(2)
# time = np.linspace(0, t, len(d.h))
# plt.plot(time, d.h, 'b', label='Height real (m)')
# plt.plot(time, m.h, 'r', label='Height model (m)')
# plt.legend(loc='best')
# plt.show()

plt.figure(2)
plt.plot(time_est, true_cost, 'b', label='Real cost')
plt.plot(time_est, model_cost, 'r', label='Model cost')
av1 = np.sum(true_cost)/len(true_cost)
av2 = np.sum(model_cost)/len(true_cost)
if av1 > av2:
    plt.ylim((0, 1e10))
else:
    plt.ylim((0, 1e10))
plt.plot(time_est, estimate_cost, 'g', label='Estimated cost')
plt.legend(loc='best')


fig, ax = plt.subplots(3)
yplot = np.array(y_real)
time = np.linspace(0, t, len(yplot))
ax[0].plot(time, yplot[:, 0], 'b', label='Real distance')
ax[1].plot(time, yplot[:, 1], 'b', label='Real elevation angle')
ax[2].plot(time, yplot[:, 2], 'b', label='Real azimuth angle')

yplot = np.array(y_model)
ax[0].plot(time, yplot[:, 0], 'r', label='Model distance')
plt.legend(loc='best')
ax[1].plot(time, yplot[:, 1], 'r', label='Model elevation angle')
ax[2].plot(time, yplot[:, 2], 'r', label='Model azimuth angle')

yplot = np.array(y_mhe)
time = np.linspace(0, t-opt.N*d.delta_t, len(yplot))
ax[0].plot(time_est, yplot[:, 0], 'g', label='Estimated distance')
ax[1].plot(time_est, yplot[:, 1], 'g', label='Estimated elevation angle')
ax[2].plot(time_est, yplot[:, 2], 'g', label='Estimated azimuth angle')
plt.legend(loc='best')
# plt.show()


fig, ax = plt.subplots(3)
yplot = np.array(d.x)
time = np.linspace(0, t, len(yplot))
ax[0].plot(time, yplot[:, 2, 0], 'b', label='Real position (r) x coordinate')
ax[1].plot(time, yplot[:, 2, 1], 'b', label='Real position (r) y coordinate')
ax[2].plot(time, yplot[:, 2, 2], 'b', label='Real position (r)')

yplot = np.array(m.Sk)
time = np.linspace(0, t, len(yplot))
ax[0].plot(time, yplot[:, 0, 0], 'r', label='Model position (r) x coordinate')
ax[1].plot(time, yplot[:, 0, 1], 'r', label='Model position (r) y coordinate')
ax[2].plot(time, yplot[:, 0, 2], 'r', label='Model position (r)')

yplot = np.array(state_estimate)
ax[0].plot(time_est, yplot[:, 0], 'g', label='Estimated position (r) x coordinate')
ax[1].plot(time_est, yplot[:, 1], 'g', label='Estimated position (r) y coordinate')
ax[2].plot(time_est, yplot[:, 2], 'g', label='Estimated position (r)')
plt.legend(loc='best')

fig, ax = plt.subplots(3)
yplot = np.array(d.x)
time = np.linspace(0, t, len(yplot))
ax[0].plot(time, yplot[:, 1, 0], 'b', label='Real velocity (v)')
ax[1].plot(time, yplot[:, 1, 1], 'b', label='Real velocity (v)')
ax[2].plot(time, yplot[:, 1, 2], 'b', label='Real velocity (v)')

yplot = np.array(m.Sk)
time = np.linspace(0, t, len(yplot))
ax[0].plot(time, yplot[:, 1, 0], 'r', label='Model velocity (v)')
ax[1].plot(time, yplot[:, 1, 1], 'r', label='Model velocity (v) ')
ax[2].plot(time, yplot[:, 1, 2], 'r', label='Model velocity (v)')

yplot = np.array(state_estimate)
ax[0].plot(time_est, yplot[:, 3], 'g', label='Estimated position (v) x coordinate')
ax[1].plot(time_est, yplot[:, 4], 'g', label='Estimated position (v) y coordinate')
ax[2].plot(time_est, yplot[:, 5], 'g', label='Estimated position (v)')
plt.legend(loc='best')

plt.show()
