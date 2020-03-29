import matplotlib.pyplot as plt
import numpy as np
import sys

from system_dynamics import dynamics
from observer import SateliteObserver
from state_model import model
from MHE_simplest import MHE
from MHE_ballisic_reg import MHE_regularisation

t_lim = 150

t = 0.00
step = int(0)  # number of measurements so saf
delta = int(0)
measurement_lapse = 0.5  # time lapse between every measurement

height = 80e3
d = dynamics(height, 22, 0, 6000, -5, 60)
o = SateliteObserver(40.24, 3.42, d)

# initialise model with errors
initialbeta = d.beta[0] + np.random.normal(0, 0.1*d.beta[0], size=1)[0]
m = model(d.r, d.v, initialbeta, measurement_lapse)
opt = MHE_regularisation(m, o, measurement_lapse)
# opt = MHE(m, o, measurement_lapse)


time = [0]
time_est = []

y_real = []
y_model = []
y_mhe = []

beta_estimation = []
state_estimate = []
real_x = []

true_cost = []
model_cost = []
estimate_cost = []

y_minus1 = o.h(d.r, 'off')

while height > 5000 and t < t_lim:
    d.step_update(d.v, d.r)
    delta = delta + 1

    # update stopping criteria
    t = t + d.delta_t
    height = d.h[len(d.h)-1]

    # measurements are only taken every 0.5 seconds (in the interest of time)
    if delta == measurement_lapse/d.delta_t:
        print(t-t_lim)
        step = step + 1
        delta = int(0)

        if step == 1:
            y_real = [o.h(d.r, 'off')]
            m.reinitialise(y_minus1, y_real[0], o, measurement_lapse)
            y_model = [o.h(m.r)]
            real_x = [[d.r, d.v]]
        else:
            m.step_update('off')  # the model is updated every 0.5 seconds (problem with discretization)
            y_real.append(o.h(d.r))

            # re-inialise model from taken measurements
            y_model.append(o.h(m.r))
            time.append(t)
            real_x.append([d.r, d.v])

        if step >= opt.N+1: # MHE is entered only when there exists sufficient measurements over the horizon
            opt.initialisation(y_real, step)
            # print(opt.cost_function(opt.x_init))

            # --------------------------------------- #
            # eps= 0.1
            # plus_eps = [[d.r[0], d.r[1], d.r[2]], [d.v[0], d.v[1], d.v[2]], opt.beta+eps]
            # minus_eps = [[d.r[0], d.r[1], d.r[2]], [d.v[0], d.v[1], d.v[2]], opt.beta-eps]
            # A = np.zeros(7)
            # B = np.zeros(7)
            # print(opt.beta)
            # A[0:3], A[3:6], a, A[6] = m.f(plus_eps[0], plus_eps[1], plus_eps[2], 'off')
            # B[0:3], B[3:6], a, B[6] = m.f(minus_eps[0], minus_eps[1], minus_eps[2], 'off')
            # derivative1 = np.zeros(7)
            # for i in range(7): derivative1[i] = (A[i]-B[i])/(2*eps)
            # plus_eps = [d.r[0], d.r[1], d.r[2], d.v[0], d.v[1], d.v[2], opt.beta+eps]
            # minus_eps = [d.r[0], d.r[1], d.r[2], d.v[0], d.v[1], d.v[2], opt.beta-eps]
            # A = opt.cost_function(plus_eps)
            # B = opt.cost_function(minus_eps)
            # derivative1 = (A-B)/(2*eps)
            # print('numerical')
            # print(derivative1)
            # print('real')
            # # print(opt.df([d.r, d.v], opt.beta)[:,6])
            # print(opt.gradient([d.r[0], d.r[1], d.r[2], d.v[0], d.v[1], d.v[2], opt.beta]))
            # print(opt.beta)
            # print(d.beta[len(d.beta)-1])
            # sys.exit()

            # plus_eps = [[d.r[0], d.r[1]+eps, d.r[2]], [d.v[0], d.v[1], d.v[2]]]
            # minus_eps = [[d.r[0], d.r[1]-eps, d.r[2]], [d.v[0], d.v[1], d.v[2]]]
            # A = opt.cost_function(plus_eps)
            # B = opt.cost_function(minus_eps)
            # derivative1 = (A - B) / (2*eps)
            # print(derivative1)
            # plus_eps = [[d.r[0], d.r[1], d.r[2]+eps], [d.v[0], d.v[1], d.v[2]]]
            # minus_eps = [[d.r[0], d.r[1], d.r[2]-eps], [d.v[0], d.v[1], d.v[2]]]
            # A = opt.cost_function(plus_eps)
            # B = opt.cost_function(minus_eps)
            # derivative1 = (A - B) / (2*eps)
            # print(derivative1)
            # sys.exit()
            # # --------------------------------------- #
            # optimisation
            print(opt.beta)
            opt.search('gradient')
            print(opt.beta)
            # store points to analyse later
            state_estimate.append(np.copy([opt.x_solution[0], opt.x_solution[1]]))
            y_mhe.append(o.h(opt.x_solution[0]))
            beta_estimation.append(opt.x_solution[2])
            # beta_estimation.append(opt.beta)
            time_est.append(t-opt.N*measurement_lapse)


            # estimate_cost.append(opt.cost_function(opt.x_solution))
            # true_cost.append(opt.cost_function(real_x[len(real_x)-1-opt.N]))
            # model_cost.append(opt.cost_function(np.array(m.Sk[len(m.Sk)-1-opt.N*opt.inter_steps])))


plt.figure(1)
time = np.linspace(0, t, len(d.beta))
plt.plot(time, d.beta, 'g', label='Real ballistic coef')
plt.plot(time_est, beta_estimation, 'r', label='Estimated ballistic coef')
# plt.plot(time, m.ballistic, 'r', label='Ballistic coef')
plt.legend(loc='best')

# plt.figure(2)
# print(estimate_cost)
# plt.plot(time_est, model_cost, 'r')
# plt.plot(time_est, estimate_cost, 'b')

# time = np.linspace(0, t, len(d.h))
# plt.plot(time, d.h, 'b', label='Height real (m)')
# plt.plot(time, m.h, 'r', label='Height model (m)')
# plt.legend(loc='best')

fig, ax = plt.subplots(3,2)
real = np.array(y_real)
yplot = np.array(y_model)
time = np.linspace(0, t, len(yplot))
ax[0, 0].plot(time, real[:, 0], 'k', label='True distance')
ax[1, 0].plot(time, real[:, 1], 'k', label='True elevation angle')
ax[2, 0].plot(time, real[:, 2], 'k', label='True azimuth angle')

ax[0, 0].plot(time, yplot[:, 0], 'r', label='Model distance')
ax[1, 0].plot(time, yplot[:, 1], 'r', label='Model elevation angle')
ax[2, 0].plot(time, yplot[:, 2], 'r', label='Model azimuth angle')

ax[0, 1].plot(time, np.abs(real[:, 0] - yplot[:, 0]), 'r', label='Model distance error')
ax[1, 1].plot(time, np.abs(real[:, 1] - yplot[:, 1]), 'r', label='Model elevation error')
ax[2, 1].plot(time, np.abs(real[:, 2] - yplot[:, 2]), 'r', label='Model azimuth error')

yplot = np.array(y_mhe)
ax[0, 0].plot(time_est, yplot[:, 0], 'b', label='Estimated distance error')
ax[1, 0].plot(time_est, yplot[:, 1], 'b', label='Estimated elevation error')
ax[2, 0].plot(time_est, yplot[:, 2], 'b', label='Estimated azimuth error')

ax[0, 1].plot(time_est, np.abs(real[0:len(yplot), 0] - yplot[:, 0]), 'b', label='Estimated distance error')
ax[1, 1].plot(time_est, np.abs(real[0:len(yplot), 1] - yplot[:, 1]), 'b', label='Estimated elevation error')
ax[2, 1].plot(time_est, np.abs(real[0:len(yplot), 2] - yplot[:, 2]), 'b', label='Estimated azimuth error')#
plt.legend(loc='best')


fig, ax = plt.subplots(3,2)
real = np.array(real_x)
yplot = np.array(m.Sk)
time = np.linspace(0, t, len(yplot))
ax[0, 0].plot(time, real[:, 0, 0], 'k', label='True position (r) x')
ax[1, 0].plot(time, real[:, 0, 1], 'k', label='True position (r) y')
ax[2, 0].plot(time, real[:, 0, 2], 'k', label='True position (r) z')

ax[0, 0].plot(time, yplot[:, 0, 0], 'r', label='Model position (r) x')
ax[1, 0].plot(time, yplot[:, 0, 1], 'r', label='Model position (r) y')
ax[2, 0].plot(time, yplot[:, 0, 2], 'r', label='Model position (r) z')

ax[0, 1].plot(time, np.abs(real[:, 0, 0] - yplot[:, 0, 0]), 'r', label='Model x error')
ax[1, 1].plot(time, np.abs(real[:, 0, 1] - yplot[:, 0, 1]), 'r', label='Model y error')
ax[2, 1].plot(time, np.abs(real[:, 0, 2] - yplot[:, 0, 2]), 'r', label='Model z error')

yplot = np.array(state_estimate)
ax[0, 0].plot(time_est, yplot[:, 0, 0], 'b', label='Estimated position (r) x')
ax[1, 0].plot(time_est, yplot[:, 0, 1], 'b', label='Estimated position (r) y')
ax[2, 0].plot(time_est, yplot[:, 0, 2], 'b', label='Estimated position (r) z')

ax[0, 1].plot(time_est, np.abs(real[0:len(yplot), 0, 0] - yplot[:, 0, 0]), 'b', label='Estimated x coordinate error')
ax[1, 1].plot(time_est, np.abs(real[0:len(yplot), 0, 1] - yplot[:, 0, 1]), 'b', label='Estimated y coordinate error')
ax[2, 1].plot(time_est, np.abs(real[0:len(yplot), 0, 2] - yplot[:, 0, 2]), 'b', label='Estimated z coordinate error')
plt.legend(loc='best')

fig, ax = plt.subplots(3,2)
yplot = np.array(m.Sk)
time = np.linspace(0, t, len(yplot))
ax[0, 0].plot(time, real[:, 1, 0], 'k', label='True velocity (v) x')
ax[1, 0].plot(time, real[:, 1, 1], 'k', label='True velocity (v) y')
ax[2, 0].plot(time, real[:, 1, 2], 'k', label='True velocity (v) z')

ax[0, 0].plot(time, yplot[:, 1, 0], 'r', label='Model velocity (v) x')
ax[1, 0].plot(time, yplot[:, 1, 1], 'r', label='Model velocity (v) y')
ax[2, 0].plot(time, yplot[:, 1, 2], 'r', label='Model velocity (v) z')

ax[0, 1].plot(time, np.abs(real[:, 1, 0] - yplot[:, 1, 0]), 'r', label='Model velocity error x')
ax[1, 1].plot(time, np.abs(real[:, 1, 1] - yplot[:, 1, 1]), 'r', label='Model velocity error y')
ax[2, 1].plot(time, np.abs(real[:, 1, 2] - yplot[:, 1, 2]), 'r', label='Model velocity error z')

yplot = np.array(state_estimate)
ax[0, 0].plot(time_est, yplot[:, 1, 0], 'b', label='Estimated velocity (v) x')
ax[1, 0].plot(time_est, yplot[:, 1, 1], 'b', label='Estimated velocity (v) y')
ax[2, 0].plot(time_est, yplot[:, 1, 2], 'b', label='Estimated velocity (v) z')

ax[0, 1].plot(time_est, np.abs(real[0:len(yplot), 1, 0] - yplot[:, 1, 0]), 'b', label='Estimated velocity error x')
ax[1, 1].plot(time_est, np.abs(real[0:len(yplot), 1, 1] - yplot[:, 1, 1]), 'b', label='Estimated velocity error y')
ax[2, 1].plot(time_est, np.abs(real[0:len(yplot), 1, 2] - yplot[:, 1, 2]), 'b', label='Estimated velocity error z')
plt.legend(loc='best')

plt.show()
