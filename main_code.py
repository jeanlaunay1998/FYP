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
o = SateliteObserver(40.24, 3.42)

# initialise model with errors
initialbeta = d.beta[0] + np.random.normal(0, 0.01*d.beta[0], size=1)[0]
m = model(d.r, d.v, initialbeta, measurement_lapse)
opt = MHE(m, o, measurement_lapse)
opt1 = MHE_regularisation(m, o, measurement_lapse)


time = [0]
time_est = []

y_real = []
y_model = []
y_mhe = []
y_mhe1 = []

beta_estimation = []
beta_estimation1 = []
state_estimate = []
state_estimate1 = []
real_x = []

true_cost = []
model_cost = []
estimate_cost = []
estimate_cost1 = []

y_minus1 = o.h(d.r, 'off')

mu1 = []
mu2 = []
R = []

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
            real_beta = [d.beta[len(d.beta)-1]]
        else:
            m.step_update('off')  # the model is updated every 0.5 seconds (problem with discretization)
            y_real.append(o.h(d.r))

            # re-inialise model from taken measurements
            y_model.append(o.h(m.r, 'off'))
            time.append(t)
            real_x.append([d.r, d.v])
            real_beta.append(d.beta[len(d.beta) - 1])

        if step >= opt.N+1: # MHE is entered only when there exists sufficient measurements over the horizon
            opt.initialisation(y_real, step)
            opt1.initialisation(y_real, step)

            opt.search('gradient')
            opt1.search('gradient')

            # --------------------------------------- #
            # opt.initialisation2(y_real, real_x, real_beta, step)
            # opt.search_coeffs()
            # mu1.append(np.copy(opt.mu1))
            # mu2.append(np.copy(opt.mu2))
            # R.append(np.copy(opt.R))

            # print(opt.cost_function(opt.x_init))

            # --------------------------------------- #
            # eps= 0.01
            # plus_eps = [[d.r[0], d.r[1], d.r[2]], [d.v[0], d.v[1], d.v[2]], opt.beta+eps]
            # minus_eps = [[d.r[0], d.r[1], d.r[2]], [d.v[0], d.v[1], d.v[2]], opt.beta-eps]
            # A = np.zeros(7)
            # B = np.zeros(7)
            # print(opt.beta)
            # A[0:3], A[3:6], a, A[6] = m.f(plus_eps[0], plus_eps[1], plus_eps[2], 'off')
            # B[0:3], B[3:6], a, B[6] = m.f(minus_eps[0], minus_eps[1], minus_eps[2], 'off')
            # derivative1 = np.zeros(7)
            # for i in range(7): derivative1[i] = (A[i]-B[i])/(2*eps)
            # plus_eps = [d.r[0]+eps, d.r[1], d.r[2], d.v[0], d.v[1], d.v[2]]#, opt.beta]
            # minus_eps = [d.r[0]-eps, d.r[1], d.r[2], d.v[0], d.v[1], d.v[2]]#, opt.beta]
            # print('A')
            # A = opt.cost_function(plus_eps)
            # print('B')
            # B = opt.cost_function(minus_eps)
            # derivative1 = (A-B)/(2*eps)
            # print('numerical')
            # print(derivative1)
            # print('real')
            # print(opt1.df([d.r, d.v], opt.beta)[:,6])
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
            # --------------------------------------- #

            # store points to analyse later
            x_iplus1 = np.copy(opt.x_solution)
            for i in range(0, opt.N):
                for j in range(opt.inter_steps):
                    x_iplus1[0], x_iplus1[1], no_interest, beta_i = m.f(x_iplus1[0], x_iplus1[1], m.beta, 'off')
            # state_estimate.append(np.copy([opt.x_solution[0], opt.x_solution[1]]))
            state_estimate.append(np.copy([x_iplus1[0], x_iplus1[1]])) # plot end of horizon

            x_iplus1 = np.copy(opt1.x_solution)
            for i in range(0, opt1.N):
                for j in range(opt1.inter_steps):
                    x_iplus1[0], x_iplus1[1], no_interest, beta_i = m.f(x_iplus1[0], x_iplus1[1], x_iplus1[2], 'off')
            # state_estimate1.append(np.copy([opt1.x_solution[0], opt1.x_solution[1]]))
            state_estimate1.append(np.copy([x_iplus1[0], x_iplus1[1]]))

            y_mhe.append(o.h(opt.x_solution[0], 'off'))
            y_mhe1.append(o.h(opt1.x_solution[0], 'off'))
            beta_estimation.append(opt.beta)
            beta_estimation1.append(opt1.x_solution[2])
            # time_est.append(t-opt.N*measurement_lapse)
            time_est.append(t)

            estimate_cost.append(opt1.cost_function([opt.x_solution[0][0],opt.x_solution[0][1],opt.x_solution[0][2],opt.x_solution[1][0],opt.x_solution[1][1],opt.x_solution[1][2], m.beta]))
            estimate_cost1.append(opt1.cost_function([opt1.x_solution[0][0],opt1.x_solution[0][1],opt1.x_solution[0][2],opt1.x_solution[1][0],opt1.x_solution[1][1],opt1.x_solution[1][2],opt1.x_solution[2]]))
            z = len(real_x)-1-opt.N
            true_cost.append(opt1.cost_function([real_x[z][0][0],real_x[z][0][1],real_x[z][0][2],real_x[z][1][0],real_x[z][1][1],real_x[z][1][2], m.beta]))
            z = len(m.Sk)-1-opt.N*opt.inter_steps
            model_cost.append(opt1.cost_function([m.Sk[z][0][0], m.Sk[z][0][1], m.Sk[z][0][2], m.Sk[z][1][0], m.Sk[z][1][1], m.Sk[z][1][2], m.beta]))

# --------------------------------------- #
# x_axis = range(len(mu1))
# plt.figure(1)
# plt.plot(x_axis, mu1)
# plt.figure(2)
# plt.plot(x_axis, mu2)
# plt.figure(3)
# yplot = np.array(R)
# plt.plot(x_axis, yplot[:, 0])
# plt.figure(4)
# plt.plot(x_axis, yplot[:, 1])
# plt.figure(5)
# plt.plot(x_axis, yplot[:, 2])
# plt.show()
# sys.exit()
# --------------------------------------- #

plt.figure(1)
time = np.linspace(0, t, len(d.beta))
plt.plot(time, d.beta, 'k', label='True system')
plt.plot(time_est, beta_estimation, 'b', label='MHE 1 (or estimation model)')
plt.plot(time_est, beta_estimation1, '--g', label='MHE 2', markersize=5)
# plt.plot(time, m.ballistic, 'r', label='Ballistic coef')
plt.legend(loc='best')

# --------------------------------------- #

plt.figure(2)
plt.plot(estimate_cost1, '--g', markersize=5, label='MHE 2')
plt.plot(model_cost, 'r', label='Estimation model')
plt.plot(estimate_cost, 'b', label='MHE 1')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Cost')

# --------------------------------------- #
fig, ax = plt.subplots(3,2)
real = np.array(y_real)
yplot = np.array(y_model)
time = np.linspace(0, t, len(yplot))
ax[0, 0].plot(time, real[:, 0], 'k')
ax[0, 0].set(xlabel='Time (s)', ylabel='d (m)')
ax[1, 0].plot(time, real[:, 1], 'k')
ax[1, 0].set(xlabel='Time (s)', ylabel='el (radians)')
ax[2, 0].plot(time, real[:, 2], 'k', label='True system')
ax[2, 0].set(xlabel='Time (s)', ylabel='az (radians)')

ax[0, 0].plot(time, yplot[:, 0], 'r')
ax[1, 0].plot(time, yplot[:, 1], 'r')
ax[2, 0].plot(time, yplot[:, 2], 'r', label='Estimation model')

ax[0, 1].plot(time, np.abs((real[:, 0] - yplot[:, 0])/real[:, 0]), 'r')
ax[1, 1].plot(time, np.abs((real[:, 1] - yplot[:, 1])/real[:, 1]), 'r')
ax[2, 1].plot(time, np.abs((real[:, 2] - yplot[:, 2])/real[:, 2]), 'r')

yplot = np.array(y_mhe)
ax[0, 0].plot(time_est, yplot[:, 0], 'b')
ax[1, 0].plot(time_est, yplot[:, 1], 'b')
ax[2, 0].plot(time_est, yplot[:, 2], 'b', label='MHE 1')

# ax[0, 1].plot(time_est, np.abs((real[0:len(yplot), 0] - yplot[:, 0])/real[0:len(yplot), 0]), 'b', label='Estimated distance error')
# ax[1, 1].plot(time_est, np.abs((real[0:len(yplot), 1] - yplot[:, 1])/real[0:len(yplot), 1]), 'b', label='Estimated elevation error')
# ax[2, 1].plot(time_est, np.abs((real[0:len(yplot), 2] - yplot[:, 2])/real[0:len(yplot), 2]), 'b', label='Estimated azimuth error')#

ax[0, 1].plot(time_est, np.abs((real[opt.N:len(real), 0] - yplot[:, 0])/real[opt.N:len(real), 0]), 'b')
ax[1, 1].plot(time_est, np.abs((real[opt.N:len(real), 1] - yplot[:, 1])/real[opt.N:len(real), 1]), 'b')
ax[2, 1].plot(time_est, np.abs((real[opt.N:len(real), 2] - yplot[:, 2])/real[opt.N:len(real), 2]), 'b')

yplot = np.array(y_mhe1)
ax[0, 0].plot(time_est, yplot[:, 0], '--g', markersize=5)
ax[1, 0].plot(time_est, yplot[:, 1], '--g', markersize=5)
ax[2, 0].plot(time_est, yplot[:, 2], '--g', label='MHE 2', markersize=5)

# ax[0, 1].plot(time_est, np.abs((real[0:len(yplot), 0] - yplot[:, 0])/real[0:len(yplot), 0]), '--+g', label='Estimated distance error')
# ax[1, 1].plot(time_est, np.abs((real[0:len(yplot), 1] - yplot[:, 1])/real[0:len(yplot), 1]), '--+', label='Estimated elevation error')
# ax[2, 1].plot(time_est, np.abs((real[0:len(yplot), 2] - yplot[:, 2])/real[0:len(yplot), 2]), '--+g', label='Estimated azimuth error')#

ax[0, 1].plot(time_est, np.abs((real[opt.N:len(real), 0] - yplot[:, 0])/real[opt.N:len(real), 0]), '--g', markersize=5)
ax[0, 1].set(xlabel='Time (s)', ylabel='Error')
ax[1, 1].plot(time_est, np.abs((real[opt.N:len(real), 1] - yplot[:, 1])/real[opt.N:len(real), 1]), '--g', markersize=5)
ax[1, 1].set(xlabel='Time (s)', ylabel='Error')
ax[2, 1].plot(time_est, np.abs((real[opt.N:len(real), 2] - yplot[:, 2])/real[opt.N:len(real), 2]), '--g', markersize=5)#
ax[2, 1].set(xlabel='Time (s)', ylabel='Error')
handles, labels = ax[2,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',  ncol=4)


# --------------------------------------- #
fig, ax = plt.subplots(3,2)
real = np.array(real_x)
yplot = np.array(m.Sk)
time = np.linspace(0, t, len(yplot))
ax[0, 0].plot(time, real[:, 0, 0], 'k')
ax[1, 0].plot(time, real[:, 0, 1], 'k')
ax[2, 0].plot(time, real[:, 0, 2], 'k', label='True system')

ax[0, 0].plot(time, yplot[:, 0, 0], 'r')
ax[1, 0].plot(time, yplot[:, 0, 1], 'r')
ax[2, 0].plot(time, yplot[:, 0, 2], 'r', label='Estimation Model')

ax[0, 1].plot(time, np.abs((real[:, 0, 0] - yplot[:, 0, 0])/real[:, 0, 0]), 'r')
ax[0, 1].set(xlabel='Time (s)', ylabel='Error')
ax[1, 1].plot(time, np.abs((real[:, 0, 1] - yplot[:, 0, 1])/real[:, 0, 1]), 'r')
ax[1, 1].set(xlabel='Time (s)', ylabel='Error')
ax[2, 1].plot(time, np.abs((real[:, 0, 2] - yplot[:, 0, 2])/real[:, 0, 2]), 'r')
ax[2, 1].set(xlabel='Time (s)', ylabel='Error')

yplot = np.array(state_estimate)
ax[0, 0].plot(time_est, yplot[:, 0, 0], 'b')
ax[1, 0].plot(time_est, yplot[:, 0, 1], 'b')
ax[2, 0].plot(time_est, yplot[:, 0, 2], 'b', label='MHE 1')

# ax[0, 1].plot(time_est, np.abs((real[0:len(yplot), 0, 0] - yplot[:, 0, 0])/real[0:len(yplot), 0, 0]), 'b', label='Estimated x coordinate error')
# ax[1, 1].plot(time_est, np.abs((real[0:len(yplot), 0, 1] - yplot[:, 0, 1])/real[0:len(yplot), 0, 1]), 'b', label='Estimated y coordinate error')
# ax[2, 1].plot(time_est, np.abs((real[0:len(yplot), 0, 2] - yplot[:, 0, 2])/real[0:len(yplot), 0, 2]), 'b', label='Estimated z coordinate error')

ax[0, 1].plot(time_est, np.abs((real[opt.N:len(real), 0, 0] - yplot[:, 0, 0])/real[opt.N:len(real), 0, 0]), 'b')
ax[1, 1].plot(time_est, np.abs((real[opt.N:len(real), 0, 1] - yplot[:, 0, 1])/real[opt.N:len(real), 0, 1]), 'b')
ax[2, 1].plot(time_est, np.abs((real[opt.N:len(real), 0, 2] - yplot[:, 0, 2])/real[opt.N:len(real), 0, 2]), 'b')

yplot = np.array(state_estimate1)
ax[0, 0].plot(time_est, yplot[:, 0, 0], '--g', markersize=5)
ax[0, 0].set(xlabel='Time (s)', ylabel='Position x (m)')
ax[1, 0].plot(time_est, yplot[:, 0, 1], '--g', markersize=5)
ax[1, 0].set(xlabel='Time (s)', ylabel='Position y (m)')
ax[2, 0].plot(time_est, yplot[:, 0, 2], '--g', label='MHE 2', markersize=5)
ax[2, 0].set(xlabel='Time (s)', ylabel='Position z (m)')

# ax[0, 1].plot(time_est, np.abs((real[0:len(yplot), 0, 0] - yplot[:, 0, 0])/real[0:len(yplot), 0, 0]), '--+g', label='Estimated x coordinate error')
# ax[1, 1].plot(time_est, np.abs((real[0:len(yplot), 0, 1] - yplot[:, 0, 1])/real[0:len(yplot), 0, 1]), '--+g', label='Estimated y coordinate error')
# ax[2, 1].plot(time_est, np.abs((real[0:len(yplot), 0, 2] - yplot[:, 0, 2])/real[0:len(yplot), 0, 2]), '--+g', label='Estimated z coordinate error')


ax[0, 1].plot(time_est, np.abs((real[opt.N:len(real), 0, 0] - yplot[:, 0, 0])/real[opt.N:len(real), 0, 0]), '--g', markersize=5)
ax[1, 1].plot(time_est, np.abs((real[opt.N:len(real), 0, 1] - yplot[:, 0, 1])/real[opt.N:len(real), 0, 1]), '--g', markersize=5)
ax[2, 1].plot(time_est, np.abs((real[opt.N:len(real), 0, 2] - yplot[:, 0, 2])/real[opt.N:len(real), 0, 2]), '--g', markersize=5)
handles, labels = ax[2,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',  ncol=4)

# --------------------------------------- #
fig, ax = plt.subplots(3,2)
yplot = np.array(m.Sk)
time = np.linspace(0, t, len(yplot))
ax[0, 0].plot(time, real[:, 1, 0], 'k')
ax[1, 0].plot(time, real[:, 1, 1], 'k')
ax[2, 0].plot(time, real[:, 1, 2], 'k', label='True system')

ax[0, 0].plot(time, yplot[:, 1, 0], 'r')
ax[1, 0].plot(time, yplot[:, 1, 1], 'r')
ax[2, 0].plot(time, yplot[:, 1, 2], 'r', label='Estimation model')

ax[0, 1].plot(time, np.abs((real[:, 1, 0] - yplot[:, 1, 0])/real[:, 1, 0]), 'r')
ax[1, 1].plot(time, np.abs((real[:, 1, 1] - yplot[:, 1, 1])/real[:, 1, 1]), 'r')
ax[2, 1].plot(time, np.abs((real[:, 1, 2] - yplot[:, 1, 2])/real[:, 1, 2]), 'r')

yplot = np.array(state_estimate)
ax[0, 0].plot(time_est, yplot[:, 1, 0], 'b')
ax[1, 0].plot(time_est, yplot[:, 1, 1], 'b')
ax[2, 0].plot(time_est, yplot[:, 1, 2], 'b', label='MHE 1')

# ax[0, 1].plot(time_est, np.abs((real[0:len(yplot), 1, 0] - yplot[:, 1, 0])/real[0:len(yplot), 1, 0]), 'b', label='Estimated velocity error x')
# ax[1, 1].plot(time_est, np.abs((real[0:len(yplot), 1, 1] - yplot[:, 1, 1])/real[0:len(yplot), 1, 1]), 'b', label='Estimated velocity error y')
# ax[2, 1].plot(time_est, np.abs((real[0:len(yplot), 1, 2] - yplot[:, 1, 2])/real[0:len(yplot), 1, 2]), 'b', label='Estimated velocity error z')

ax[0, 1].plot(time_est, np.abs((real[opt.N:len(real), 1, 0] - yplot[:, 1, 0])/real[opt.N:len(real), 1, 0]), 'b')
ax[1, 1].plot(time_est, np.abs((real[opt.N:len(real), 1, 1] - yplot[:, 1, 1])/real[opt.N:len(real), 1, 1]), 'b')
ax[2, 1].plot(time_est, np.abs((real[opt.N:len(real), 1, 2] - yplot[:, 1, 2])/real[opt.N:len(real), 1, 2]), 'b')


yplot = np.array(state_estimate1)
ax[0, 0].plot(time_est, yplot[:, 1, 0], '--g', markersize=5)
ax[0, 0].set(xlabel='Time (s)', ylabel='Velocity x (m/s)')
ax[1, 0].plot(time_est, yplot[:, 1, 1], '--g', markersize=5)
ax[1, 0].set(xlabel='Time (s)', ylabel='Velocity y (m/s)')
ax[2, 0].plot(time_est, yplot[:, 1, 2], '--g', label='MHE 2', markersize=5)
ax[2, 0].set(xlabel='Time (s)', ylabel='Velocity z (m/s)')

# ax[0, 1].plot(time_est, np.abs((real[0:len(yplot), 1, 0] - yplot[:, 1, 0])/real[0:len(yplot), 1, 0]), '--g', label='Estimated velocity error x')
# ax[1, 1].plot(time_est, np.abs((real[0:len(yplot), 1, 1] - yplot[:, 1, 1])/real[0:len(yplot), 1, 1]) , '--g', label='Estimated velocity error y')
# ax[2, 1].plot(time_est, np.abs((real[0:len(yplot), 1, 2] - yplot[:, 1, 2])/real[0:len(yplot), 1, 2]) , '--g', label='Estimated velocity error z')


ax[0, 1].plot(time_est, np.abs((real[opt.N:len(real), 1, 0] - yplot[:, 1, 0])/real[opt.N:len(real), 1, 0]), '--g', markersize=5)
ax[0, 1].set(xlabel='Time (s)', ylabel='Error')
ax[1, 1].plot(time_est, np.abs((real[opt.N:len(real), 1, 1] - yplot[:, 1, 1])/real[opt.N:len(real), 1, 1]), '--g', markersize=5)
ax[1, 1].set(xlabel='Time (s)', ylabel='Error')
ax[2, 1].plot(time_est, np.abs((real[opt.N:len(real), 1, 2] - yplot[:, 1, 2])/real[opt.N:len(real), 1, 2]), '--g', markersize=5)
ax[2, 1].set(xlabel='Time (s)', ylabel='Error')
handles, labels = ax[2,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',  ncol=4)
plt.show()
