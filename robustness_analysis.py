import numpy.linalg as LA
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from extendedKF import ExtendedKalmanFilter
import sys

from system_dynamics import dynamics
from observer import SateliteObserver
from state_model import model
from MHE_ballisic_reg import MHE_regularisation
from MHE_ballistic_MS import total_ballistic
from multi_shooting_MHE import multishooting
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from MS_MHE_PE import MS_MHE_PE
from memory import Memory
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns


t_lim = 130
measurement_lapse = 0.5  # time lapse between every measurement
t = 0.00
step = int(0)  # number of measurements measurements made
delta = int(0)
height = 80e3

# Initialisation of true dynamics and approximation model
o = SateliteObserver(22, 10)
d = dynamics(height, 22, 0, 6000, -5, 60, o, wind='on', mass_change='on')
initialbeta = d.beta[0] + np.random.normal(0, 0.01*d.beta[0], size=1)[0]
m = model(d.r, d.v, initialbeta, measurement_lapse)


# covariance matrices initialise with the ideal case
R = np.array([[100**2, 0, 0], [0, (1e-3)**2, 0], [0, 0, (1e-3)**2]]) # Measurement covariance matrix
P0 = np.zeros((7,7))  # Initial covariance matrix
Q = np.zeros((7,7))  # Process noise covariance matrix
qa = 0.75 #  Estimated deviation of acceleration between real state and approximated state
for i in range(3):
    P0[i,i] = 500**2
    P0[i+3, i+3] = 1000**2

    Q[i, i] = (qa*measurement_lapse**3)/3
    Q[i, i+3] = (qa*measurement_lapse**2)/2
    Q[i+3, i] = (qa*measurement_lapse**2)/2
    Q[i+3, i+3] = qa*measurement_lapse
P0[6,6] = 20**2
Q[6,6] = 60

time = [0]
y_real = []
y_model = []
real_x = []
model_x = []
y_minus1 = o.h(d.r, 'off')

y_error = []
process_error = []


while height > 5000 and t < t_lim:
    d.step_update(d.v, d.r)
    delta = delta + 1

    # update stopping criteria
    t = t + d.delta_t
    height = d.h[len(d.h)-1]

    # measurements are only taken every 0.5 seconds (in the interest of time)
    if delta == measurement_lapse/d.delta_t:
        print('time: ', t-t_lim)
        step = step + 1
        delta = int(0)

        if step == 1:
            y_real = [o.h(d.r, 'off')]
            m.reinitialise(y_minus1, y_real[0], o, measurement_lapse)

            y_model = [o.h(m.r)]
            real_x = [[d.r, d.v]]
            model_x = [[m.r[0], m.r[1], m.r[2], m.v[0], m.v[1],m.v[2], m.beta]]
            real_beta = [d.beta[len(d.beta)-1]]

            # initial error not that the process error is yet not considered as it the error in the
            # state at step 0 corresponds to the initialisation error
            y_error.append(np.array(y_real[0]) - np.array(o.h(d.r, 'off')))
            previous_state = [d.r[0], d.r[1], d.r[2], d.v[0], d.v[1], d.v[2], d.beta[len(d.beta)-1][0]]

        else:
            m.step_update('off')  # the model is updated every 0.5 seconds (problem with discretization)
            y_real.append(o.h(d.r))

            # re-initialise model from taken measurements
            y_model.append(o.h(m.r, 'off'))
            time.append(t)
            real_x.append([d.r, d.v])
            model_x.append([m.r[0], m.r[1],m.r[2], m.v[0], m.v[1],m.v[2]])
            real_beta.append(d.beta[len(d.beta) - 1])

            # errors
            y_error.append(np.array(y_real[step-1]) - np.array(o.h(d.r, 'off')))
            process_error.append((np.array([d.r[0], d.r[1], d.r[2], d.v[0], d.v[1], d.v[2], d.beta[len(d.beta)-1][0]]) - m.f(previous_state, 'off')))
            previous_state = [d.r[0], d.r[1], d.r[2], d.v[0], d.v[1], d.v[2], d.beta[len(d.beta)-1][0]]

y_mean = np.sum(y_error, 0)/len(y_error)
process_mean = np.sum(np.abs(process_error), 0)/len(process_error)

y_covariance = np.zeros((3,3))
for i in range(len(y_error)):
    diff = y_mean - y_error[i]
    update = np.zeros((3,3))
    for j in range(3):
        update[j, :] = diff[j]*diff
    y_covariance = y_covariance + update/len(y_error)

process_covariance = np.zeros((7,7))
for i in range(len(process_error)):
    diff = process_mean - process_error[i]
    update = np.zeros((7,7))
    for j in range(7):
        update[j, :] = diff[j]*diff
    process_covariance = process_covariance + update/len(process_error)
print(y_mean)
print(process_mean)
print('----')
print(y_covariance)
print('----')
print(process_covariance)


sns.set()

theoretical_r = np.random.normal(0, Q[0,0]**0.5, size=50000)
theoretical_v = np.random.normal(0, Q[3,3]**0.5, size=50000)
theoretical_beta = np.random.normal(0, Q[6,6]**0.5, size=50000)

fig, axes = plt.subplots(1,3)

prob = pd.DataFrame(np.array(process_error)[:,0:3])
prob.plot(kind="kde", ax=axes[0])
prob = pd.DataFrame(theoretical_r)
prob.plot(kind="kde", style='--', color='k', ax=axes[0])
axes[0].axvline(np.sum(process_mean[0:3])/3, color='k', linestyle='--')

prob = pd.DataFrame(np.array(process_error)[:,3:6])
prob.plot(kind="kde", ax=axes[1])
prob = pd.DataFrame(theoretical_v)
prob.plot(kind="kde", style='--', color='k', ax=axes[1])
axes[1].axvline(sum(process_mean[3:6])/3, color='k', linestyle='--')

prob = pd.DataFrame(np.array(process_error)[:,6])
prob.plot(kind="kde", ax=axes[2])
prob = pd.DataFrame(theoretical_beta)
prob.plot(kind="kde", style='--', color='k', ax=axes[2])
axes[2].axvline(process_mean[6], color='k', linestyle='--')



axes[0].set_xlabel(r'$r \; (km)$')
axes[0].legend([r"$x$",r"$y$",r"$z$"]);
axes[1].set_xlabel(r'$v \; (m.s^{-1})$')
axes[1].legend([r"$x$",r"$y$",r"$z$"]);
axes[2].set_xlabel(r'$\beta \; (kg.m^{-2})$')
axes[2].legend([r"$\beta$"]);
plt.subplots_adjust(left=0.05, right=0.97, top=0.75, bottom=0.25)
plt.show()
sys.exit()



#  ----------------------------------------------------------------------------------------  #


error = [-0.50, -0.25, -0.1, 0, 0.1, 0.25, 0.5]
ukf_errors = []
ekf_errors = []
balreg_error = []
total_reg = []
multi_reg = []

Q_0 = Q
P0_0 = P0
R_0 = R

for s in range(len(error)):
    N = [20, 20]  # size of the horizon
    t = 0.00
    step = int(0)  # number of measurements measurements made
    delta = int(0)

    # covariance matrices
    R = R_0 + error[s]*R
    P0 = P0_0 #+ error[s]*P0
    Q = Q_0 # + error[s]*Q

    # Initialisation of estimators
    opt = []
    MHE_type = ['Ballistic reg', 'Multi-shooting']
    method = ['Newton LS', 'Newton LS', 'Newton LS']
    measurement_pen = []
    model_pen = []
    arrival = [1, 1, 1]
    for i in range(len(N)):
        if MHE_type[i] == 'Total ballistic':
            opt.append(total_ballistic(m, o, N[i], measurement_lapse, model_pen, method[i], Q, R))
        elif MHE_type[i] == 'Ballistic reg':
            opt.append(MHE_regularisation(m, o, N[i], measurement_lapse, model_pen, method[i], Q, R))
        elif MHE_type[i] == 'Multi-shooting':
            opt.append(multishooting(m, d, o, N[i], measurement_lapse, measurement_pen, model_pen, Q, R, arrival[i], method[i]))
        elif MHE_type[i] == 'MS with PE':
            opt.append(MS_MHE_PE(m, d, o, N[i], measurement_lapse, measurement_pen, model_pen, [P0, Q, R], opt_method=method[i]))
        else:
            print('Optimization type not recognize')
            sys.exit()
    memory = Memory(o, N, len(N), MHE_type)

    # unscented kalman filter
    points = MerweScaledSigmaPoints(n=7, alpha=.1, beta=2., kappa=-1)
    ukf = UKF(dim_x=7, dim_z=3, fx=m.f, hx=o.h, dt=measurement_lapse, points=points)
    ukf.P = P0
    ukf.Q = Q
    ukf.R = R

    # extended kalman filter
    ekf = ExtendedKalmanFilter(dim_x=7, dim_z=3, dim_u=0)
    ekf.P = P0
    ekf.Q = Q
    ekf.R = R


    while height > 5000 and t < t_lim:
        delta = delta + 1
        t = t + d.delta_t

        # measurements are only taken every 0.5 seconds (in the interest of time)
        if delta == measurement_lapse/d.delta_t:
            print('time: ', t-t_lim)
            step = step + 1
            delta = int(0)

            if step == 1:

                ukf.x = np.array(model_x[0])
                ekf.x = np.array(model_x[0])
                UKF_state = [np.copy(ukf.x)]
                EKF_state = [np.copy(ekf.x)]
            else:

                ukf.predict()
                ekf.F = opt[0].dfdx(ekf.x)
                ekf.predict(fx=m.f)
                ukf.update(y_real[step-1]) # change needed
                ekf.update(z=y_real[step-1], HJacobian=opt[0].dh, Hx=o.h) # change needed
                UKF_state.append(np.copy(ukf.x))
                EKF_state.append(np.copy(ekf.x))


            for i in range(len(opt)):
                if step >= opt[i].N+1: # MHE is entered only when there exists sufficient measurements over the horizon
                    if step==opt[i].N+1:
                        opt[i].estimator_initilisation(step, y_real)
                    else:
                        opt[i].slide_window(y_real[step-1])
                    print(MHE_type[i])
                    opt[i].estimation()
                    if MHE_type[i] == 'Total ballistic' or MHE_type[i] == 'Ballistic reg':
                        x = opt[i].last_state()
                        memory.save_data(t, x, o.h(m.r, 'off'), opt[i].cost(opt[i].vars), i)
                    else:
                        memory.save_data(t, opt[i].vars, o.h(m.r, 'off'), opt[i].cost(opt[i].vars), i)

    a,b,c,f  =  memory.make_plots(real_x, real_beta, y_real, m.Sk, UKF_state, EKF_state)
    print(a)
    print(b)
    print(c)
    print(f)
    print('---------------------------------------------------')
    ekf_errors.append(a)
    ukf_errors.append(b)
    balreg_error.append(c)
    multi_reg.append(f)
    # totalreg.append(e)

print(ekf_errors)
print(ukf_errors)
print(balreg_error)
print(total_reg)
print(multi_reg)

fig, ax = plt.subplots(3,1)

for j in range(3):
    y_plot = []
    for i in range(len(error)):
        y_plot.append(np.sum(np.power(ekf_errors[i][j], 2))**0.5)
    ax[j].plot(np.multiply(error, 100), y_plot,'r', label='EKF')

    y_plot = []
    for i in range(len(error)):
        y_plot.append(np.sum(np.power(ukf_errors[i][j], 2))**0.5)
    ax[j].plot(np.multiply(error, 100), y_plot, '--r', label='UKF')

    y_plot = []
    for i in range(len(error)):
        y_plot.append(np.sum(np.power(multi_reg[i][j], 2)) ** 0.5)
    ax[j].plot(np.multiply(error, 100), y_plot, 'b', label='Ballistic reg MHE')

    y_plot = []
    for i in range(len(error)):
        y_plot.append(np.sum(np.power(balreg_error[i][j], 2)) ** 0.5)
    ax[j].plot(np.multiply(error, 100), y_plot, '--b', label='Multi-shooting MHE')

plot_type = [r"$||r - \hat{r}||$", r"$||v - \hat{v}||$", r"$||\beta - \hat{\beta}||$" ]
i = 0
for axs in ax.flat:
    axs.set(xlabel=r"$e_{R} \; (\%)$", ylabel=plot_type[i])
    i = i +1
for axs in ax.flat:
    axs.label_outer()
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)


# ax[0].set(xlabel=r"$e_{R} \; (\%)$", ylabel=r"$||r - \hat{r}||")
# ax[1].set(xlabel=r"$e_{R} \; (\%)$", ylabel=r"$||v - \hat{v}||")
# ax[2].set(xlabel=r"$e_{R} \; (\%)$", ylabel=r"$||\beta - \hat{\beta}||")

# fig, ax = plt.subplots(3,1)
# for j in range(3):
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(ekf_errors[i][0][j])
#     ax[j].plot(error, y_plot,'r', label='EKF')
#
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(ukf_errors[i][0][j])
#     ax[j].plot(error, y_plot, '--r', label='UKF')
#
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(multi_reg[i][0][j])
#     ax[j].plot(error, y_plot, 'b', label='Ballistic reg MHE')
#
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(balreg_error[i][0][j])
#     ax[j].plot(error, y_plot, '--b', label='Multi-shooting MHE')
# handles, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=4)
#
#
# fig, ax = plt.subplots(3,1)
# for j in range(3):
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(ekf_errors[i][1][j])
#     ax[j].plot(error, y_plot, 'r', label='EKF')
#
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(ukf_errors[i][1][j])
#     ax[j].plot(error, y_plot, '--r', label='UKF')
#
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(multi_reg[i][1][j])
#     ax[j].plot(error, y_plot, 'b', label='Ballistic reg MHE')
#
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(balreg_error[i][1][j])
#     ax[j].plot(error, y_plot, '--b', label='Multi-shooting MHE')
# handles, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=4)
# plt.figure()
# for j in range(1):
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(ekf_errors[i][2][j])
#     plt.plot(error, y_plot,'r', label='EKF')
#
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(ukf_errors[i][2][j])
#     plt.plot(error, y_plot,'--r', label='UKF')
#
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(multi_reg[i][2][j])
#     plt.plot(error, y_plot, 'b', label='Ballistic reg MHE')
#
#     y_plot = []
#     for i in range(len(error)):
#         y_plot.append(balreg_error[i][1][j])
#     plt.plot(error, y_plot, '--b', label='Multi-shooting MHE')
# plt.legend(loc='best')
plt.show()