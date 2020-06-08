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
measurement_lapse = 0.5 # time lapse between every measurement
t = 0.00
step = int(0)  # number of measurements measurements made
delta = int(0)
height = 80e3

# Initialisation of true dynamics and approximation model
model_error = True

ekf_conv = []
ukf_conv = []
balreg_conv = []
multi_reg_conv = []

if model_error:
    print('----------------------------------------------------') # 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125

    R_m = [np.array([[100 ** 2, 0, 0], [0, (0.5e-3) ** 2, 0], [0, 0, (0.5e-3) ** 2]]),\
          np.array([[100 ** 2, 0, 0], [0, (1e-3) ** 2, 0], [0, 0, (1e-3) ** 2]]),\
          [np.array([[100 ** 2, 0, 0], [0, (2.5e-3) ** 2, 0], [0, 0, (2.5e-3) ** 2]])],\
          [np.array([[100 ** 2, 0, 0], [0, (5e-3) ** 2, 0], [0, 0, (5e-3) ** 2]])],\
          [np.array([[100 ** 2, 0, 0], [0, (7.5e-3) ** 2, 0], [0, 0, (7.5e-3) ** 2]])],\
          [np.array([[100 ** 2, 0, 0], [0, (10e-3) ** 2, 0], [0, 0, (10e-3) ** 2]])],\
           [np.array([[100 ** 2, 0, 0], [0, (12.5e-3) ** 2, 0], [0, 0, (12.5e-3) ** 2]])]]
    a_res = [1.5, 3, 4.5, 6, 9, 12]
    qqa = [0.1, 0.5, 2, 4, 20, 42]
    beta_res = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.08]
    qbeta = [20, 40, 80, 190, 1000, 1800, 180*200]


    Nruns = len(R_m)
    ukf_errors = []
    ekf_errors = []
    balreg_error = []
    total_reg = []
    multi_reg = []

    for s in range(len(R_m)):
        N = [20, 20]  # size of the horizon
        MHE_type = ['Multi-shooting', 'Ballistic reg', 'Multi-shooting']
        method = ['Newton LS', 'Newton LS']
        arrival = [1, 1]

        t = 0.00
        step = int(0)  # number of measurements measurements made
        delta = int(0)
        height = 80e3

        # Initialisation of true dynamics and approximation model
        # Initialisation of true dynamics and approximation model
        R = R_m[s]
        print(R)
        o = SateliteObserver(22, 10, R)
        d = dynamics(height, 22, 0, 6000, -5, 60, o, wind='off', mass_change='off', q_a=9, q_bp=0.01)
        initialbeta = d.beta[0] + np.random.normal(0, 0.01 * d.beta[0], size=1)[0]
        m = model(d.r, d.v, initialbeta, measurement_lapse)

        # covariance matrices
        P0 = np.zeros((7, 7))  # Initial covariance matrix
        Q = np.zeros((7, 7))  # Process noise covariance matrix
        qa = 0.3 # qqa[s]  # 0.75 #  Estimated deviation of acceleration between real state and approximated state

        for i in range(3):
            P0[i, i] = 500 ** 2
            P0[i + 3, i + 3] = 1000 ** 2

            Q[i, i] = (qa * measurement_lapse ** 3) / 3
            Q[i, i + 3] = (qa * measurement_lapse ** 2) / 2
            Q[i + 3, i] = (qa * measurement_lapse ** 2) / 2
            Q[i + 3, i + 3] = qa * measurement_lapse
        P0[6, 6] = 20 ** 2
        Q[6, 6] = 20

        # Initialisation of estimators
        opt = []
        measurement_pen = []
        model_pen = []
        model_pen = [1e-3, 1e-3, 1e-3, 5e-1, 5e-1, 5e-1,1e-2]

        for i in range(len(N)):
            if MHE_type[i] == 'Total ballistic':
                opt.append(total_ballistic(m, o, N[i], measurement_lapse, model_pen, method[i], Q, R, arrival[i]))
            elif MHE_type[i] == 'Ballistic reg':
                opt.append(MHE_regularisation(m, o, N[i], measurement_lapse, model_pen, method[i], Q, R, arrival[i]))
            elif MHE_type[i] == 'Multi-shooting':
                opt.append(multishooting(m, d, o, N[i], measurement_lapse, measurement_pen, model_pen, Q, R, arrival[i],
                                         method[i]))
            elif MHE_type[i] == 'MS with PE':
                opt.append(MS_MHE_PE(m, d, o, N[i], measurement_lapse, measurement_pen, model_pen, [P0, Q, R],
                                     opt_method=method[i]))
            else:
                print('MHE type not recognize')
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

        time = [0]
        y_real = []
        y_model = []
        real_x = []
        y_minus1 = o.h(d.r, 'off')

        while height > 5000 and t < t_lim:
            d.step_update(d.v, d.r)
            delta = delta + 1

            # update stopping criteria
            t = t + d.delta_t
            height = d.h[len(d.h) - 1]

            # measurements are only taken every 0.5 seconds (in the interest of time)
            if delta == measurement_lapse / d.delta_t:
                print('time: ', t - t_lim)
                step = step + 1
                delta = int(0)

                if step == 1:
                    y_real = [o.h(d.r, 'off')]
                    m.reinitialise(y_minus1, y_real[0], o, measurement_lapse)

                    y_model = [o.h(m.r)]
                    real_x = [[d.r, d.v]]
                    real_beta = [d.beta[len(d.beta) - 1]]

                    ukf.x = np.array([m.r[0], m.r[1], m.r[2], m.v[0], m.v[1], m.v[2], m.beta])
                    ekf.x = np.array([m.r[0], m.r[1], m.r[2], m.v[0], m.v[1], m.v[2], m.beta])
                    UKF_state = [np.copy(ukf.x)]
                    EKF_state = [np.copy(ekf.x)]

                else:
                    m.step_update('off')  # the model is updated every 0.5 seconds (problem with discretization)
                    y_real.append(o.h(d.r))

                    # re-initialise model from taken measurements
                    y_model.append(o.h(m.r, 'off'))
                    time.append(t)
                    real_x.append([d.r, d.v])
                    real_beta.append(d.beta[len(d.beta) - 1])

                    ukf.predict()
                    ekf.F = opt[0].dfdx(ekf.x)
                    ekf.predict(fx=m.f)
                    ukf.update(y_real[len(y_real) - 1])
                    ekf.update(z=y_real[len(y_real) - 1], HJacobian=opt[0].dh, Hx=o.h)
                    UKF_state.append(np.copy(ukf.x))
                    EKF_state.append(np.copy(ekf.x))

                for i in range(len(opt)):
                    if step >= opt[
                        i].N + 1:  # MHE is entered only when there exists sufficient measurements over the horizon
                        if step == opt[i].N + 1:
                            opt[i].estimator_initilisation(step, y_real)
                        else:
                            opt[i].slide_window(y_real[step - 1])
                        opt[i].estimation()
                        if MHE_type[i] == 'Total ballistic' or MHE_type[i] == 'Ballistic reg':
                            x = opt[i].last_state()
                            memory.save_data(t, x, o.h(m.r, 'off'), opt[i].cost(opt[i].vars), i)
                        else:
                            memory.save_data(t, opt[i].vars, o.h(m.r, 'off'), opt[i].cost(opt[i].vars), i)
        a, b, c, f = memory.make_absolute_plots(real_x, real_beta, y_real, m.Sk, UKF_state, EKF_state)
        # beta_plot.append(real_beta)
        print(a)
        print(b)
        print(c)
        print(f)
        print('---------------------------------------------------')
        ekf_errors.append(a[0])
        ekf_conv.append([a[1], a[2]])
        ukf_errors.append(b[0])
        ukf_conv.append([b[1], b[2]])
        balreg_error.append(c[0])
        balreg_conv.append([c[1], c[2]])
        multi_reg.append(f[0])
        multi_reg_conv.append([f[1], f[2]])


print(ekf_errors)
print(ukf_errors)
print(balreg_error)
print(total_reg)
print(multi_reg)

fig, ax = plt.subplots(3,1)
for j in range(3):
    y_plot = []
    for i in range(Nruns):
        y_plot.append(np.sum(np.power(ekf_errors[i][j], 2))**0.5)
    ax[j].plot(np.power(np.array(R_m)[:, 1,1], 0.5), y_plot,'r', label='EKF')

    y_plot = []
    for i in range(Nruns):
        y_plot.append(np.sum(np.power(ukf_errors[i][j], 2))**0.5)
    ax[j].plot(np.power(np.array(R_m)[:, 1,1], 0.5), y_plot, '--r', label='UKF')

    y_plot = []
    for i in range(Nruns):
        y_plot.append(np.sum(np.power(multi_reg[i][j], 2)) ** 0.5)
    ax[j].plot(np.power(np.array(R_m)[:, 1,1], 0.5), y_plot, 'b', label='Ballistic reg MHE')

    y_plot = []
    for i in range(Nruns):
        y_plot.append(np.sum(np.power(balreg_error[i][j], 2)) ** 0.5)
    ax[j].plot(np.power(np.array(R_m)[:, 1,1], 0.5), y_plot, '--b', label='Multi-shooting MHE')

plot_type = [r"$||r - \hat{r}||$", r"$||v - \hat{v}||$", r"$||\beta - \hat{\beta}||$" ]
i = 0
for axs in ax.flat:
    axs.set(xlabel=r"$\sigma_\beta \; (rad)$", ylabel=plot_type[i])
    i = i +1
for axs in ax.flat:
    axs.label_outer()
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)

plt.figure()
plt.plot(np.power(np.array(R_m)[:, 1,1], 0.5), 100*np.array(ekf_conv)[:, 0],'r', label='EKF')
plt.plot(np.power(np.array(R_m)[:, 1,1], 0.5), 100*np.array(ekf_conv)[:, 0],'--r', label='UKF')
plt.plot(np.power(np.array(R_m)[:, 1,1], 0.5), 100*np.array(balreg_conv)[:, 0],'b', label='Ballistic reg MHE')
plt.plot(np.power(np.array(R_m)[:, 1,1], 0.5), 100*np.array(multi_reg_conv)[:, 0],'--b', label='Multi-shooting MHE')
plt.xlabel(r"$\sigma_\beta \; (rad)$")
plt.ylabel("Non-divergence (%)")
# plt.legend(loc='upper center', ncol=4)

plt.show()