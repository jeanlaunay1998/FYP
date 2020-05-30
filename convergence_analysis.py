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
from newton_method import conv_analysis
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from MS_MHE_PE import MS_MHE_PE
from memory import Memory


import matplotlib.pyplot as plt
import seaborn as sns

t_lim = 9
N = [2, 2, 2]  # size of the horizon
measurement_lapse = 0.5  # time lapse between every measurement
stop_points = [2, 4, 6, 8] # [20, 65, 90, 120] #

t = 0.00
step = int(0)  # number of measurements measurements made
delta = int(0)
height = 80e3
# Initialisation of true dynamics and approximation model
o = SateliteObserver(22, 10)  #40.24, 3.42)
d = dynamics(height, 22, 0, 6000, -5, 60, o, 'off', 'off')
initialbeta = d.beta[0] + np.random.normal(0, 0.01*d.beta[0], size=1)[0]
m = model(d.r, d.v, initialbeta, measurement_lapse)


# covariance matrices
R = np.array([[50**2, 0, 0], [0, (1e-3)**2, 0], [0, 0, (1e-3)**2]]) # Measurement covariance matrix
P0 = np.zeros((7,7))  # Initial covariance matrix
Q = np.zeros((7,7))  # Process noise covariance matrix
qa = 5 #  Estimated deviation of acceleration between real state and approximated state

for i in range(3):
    P0[i,i] = 500**2
    P0[i+3, i+3] = 1000**2

    Q[i, i] = (qa*measurement_lapse**3)/3
    Q[i, i+3] = (qa*measurement_lapse**2)/2
    Q[i+3, i] = (qa*measurement_lapse**2)/2
    Q[i+3, i+3] = qa*measurement_lapse
P0[6,6] = 20**2
Q[6,6] = 100

# Initialisation of estimators
opt = []
MHE_type = [ 'Ballistic reg', 'Hybrid multi-shooting', 'Multi-shooting']
method = ['Newton LS', 'Newton LS','Newton LS']
# measurement_pen =  [0.06, 75, 75] # coefficients obtained from the estimation opt of MS_MHE_PE
# model_pen =  [1e3, 1e3, 1e3, 1e1, 1e1, 1e1, 0.411]  # coefficients obtained from the estimation opt of MS_MHE_PE
measurement_pen =  [1e6, 1e1, 1e1]  # [1e7, 1, 1] #  [1e6, 1e-1, 1e-1] # [0.06, 80, 80] [1, 1e2, 1e3]  #
model_pen =  [1e-3,1e-3,1e-3, 5e-1,5e-1,5e-1, 1e-2] # [1e6, 1e6, 1e6, 1e1, 1e1, 1e1, 1e-1] #  [3, 3, 3, 1, 1, 1, 0.43] #[1, 1, 1, 1e1, 1e1, 1e1, 1e-1]
arrival = [1, 1, 1]

for i in range(len(N)):
    if MHE_type[i] == 'Hybrid multi-shooting':
        opt.append(total_ballistic(m, o, N[i], measurement_lapse, model_pen, method[i]))
    elif MHE_type[i] == 'Ballistic reg':
        opt.append(MHE_regularisation(m, o, N[i], measurement_lapse, model_pen, method[i]))
    elif MHE_type[i] == 'Multi-shooting':
        opt.append(multishooting(m, d, o, N[i], measurement_lapse, measurement_pen, model_pen, Q, R, arrival[i], method[i]))
    elif MHE_type[i] == 'MS with PE':
        opt.append(MS_MHE_PE(m, d, o, N[i], measurement_lapse, measurement_pen, model_pen, [P0, Q, R], opt_method=method[i]))
    else:
        print('Optimization type not recognize')
        sys.exit()


memory = Memory(o, N, len(N))

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
penalties = []

cost_history = []
x_history = []
for i in range(len(stop_points)):
    cost_history.append([])
    x_history.append([])
k = 0

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
            real_beta = [d.beta[len(d.beta)-1]]


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
            if step >= opt[i].N+1: # MHE is entered only when there exists sufficient measurements over the horizon
                if step==opt[i].N+1:
                    opt[i].estimator_initilisation(step, y_real)
                    # opt[i].vars = opt[i].vars * 1.01
                else:
                    opt[i].slide_window(y_real[step-1])
                # for j in range(len(opt[i].vars)):
                #     opt[i].vars[j] = opt[i].vars[j]*(1+ np.random.normal(0, 0.1, 1)[0])
                if k < len(stop_points):
                    if int(t) == stop_points[k]:
                        print('i:', i)
                        print('k:', k)
                        z, c = conv_analysis(opt[i].vars, opt[i].gradient, opt[i].hessian, opt[i].cost, 'on')
                        cost_history[k].append(c)
                        x_history[k].append(z)
                        if i == len(opt)-1: k = k+1
                        print('k:', k)
                opt[i].estimation()


sns.set()
fig, ax = plt.subplots(len(stop_points)//2, 2)

for i in range(len(stop_points)):
    for j in range(len(opt)):
        y = np.array(cost_history[i])[j] - np.min(np.array(cost_history[i])[j])
        # for l in range(len(y)):
        #     if y[l] == 0:
        #         y[l] = y[l-1]
        #     else:
        #         y[l] = np.log10(y[l])
        # y = np.divide()
        ax[i//2, i%2].plot(y,  label=MHE_type[j])
    ax[i//2, i%2].set_title('Time (s) = ' +  str(stop_points[i]))
    ax[i//2, i%2].set_yscale("log")

for axs in ax.flat:
    axs.set(xlabel='Iterations', ylabel=r'$C_i - C_{f}$')
for axs in ax.flat:
    axs.label_outer()

handles, labels = ax[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)


# change in state between iterations
colors= ['b', 'r', 'g', 'y']
shape = ['-+', '--', '-s']

fig1, ax1 = plt.subplots(len(stop_points)//2, 2)
fig2, ax2 = plt.subplots(len(stop_points)//2, 2)
fig3, ax3 = plt.subplots(len(stop_points)//2, 2)

for i in range(len(stop_points)):
    for j in range(len(opt)):
        r_change = []
        v_change = []
        beta_change = []
        for k in range(len(x_history[i][j])-1):
            if len(x_history[i][j][k]) == 7:
                r_change.append(np.power(np.sum(np.power(np.array(x_history[i][j][k+1])[0:3] - np.array(x_history[i][j][k])[0:3], 2)), 0.5))
                v_change.append(np.power(np.sum(np.power(np.array(x_history[i][j][k+1])[3:6] - np.array(x_history[i][j][k])[3:6], 2)), 0.5))
                beta_change.append(np.sum(np.abs(np.array(x_history[i][j][k+1])[6] - np.array(x_history[i][j][k])[6])))
            if len(x_history[i][j][k]) == 7 + opt[j].N:
                r_change.append(np.power(np.sum(np.power(np.array(x_history[i][j][k+1])[0:3] - np.array(x_history[i][j][k])[0:3], 2)), 0.5))
                v_change.append(np.power(np.sum(np.power(np.array(x_history[i][j][k+1])[3:6] - np.array(x_history[i][j][k])[3:6], 2)), 0.5))
                beta_change.append(np.sum(np.abs(np.array(x_history[i][j][k+1])[6:len(x_history[i][j][k])] - np.array(x_history[i][j][k])[6:len(x_history[i][j][k])]))/(opt[j].N+1))
            if len(x_history[i][j][k]) == 7*(1+opt[j].N):
                r_change.append(np.power(np.sum(np.power(np.array(x_history[i][j][k+1])[0:3] - np.array(x_history[i][j][k])[0:3], 2)), 0.5)/((opt[j].N +1)))
                v_change.append(np.power(np.sum(np.power(np.array(x_history[i][j][k+1])[3:6] - np.array(x_history[i][j][k])[3:6], 2)), 0.5)/((opt[j].N +1)))
                beta_change.append(np.sum(np.abs(np.array(x_history[i][j][k+1])[6] - np.array(x_history[i][j][k])[6]))/(opt[j].N +1))
                for n in range(1, opt[j].N):
                    r_change[len(r_change)-1] = r_change[len(r_change)-1] + np.power(np.sum(np.power(np.array(x_history[i][j][k+1])[n*7:n*7+3] - np.array(x_history[i][j][k])[n*7:n*7+3], 2)), 0.5)/((opt[j].N +1))
                    v_change[len(r_change)-1] = v_change[len(r_change)-1] + np.power(np.sum(np.power(np.array(x_history[i][j][k+1])[n*7+3:n*7+6] - np.array(x_history[i][j][k])[n*7+3:n*7+6], 2)), 0.5)/((opt[j].N +1))
                    beta_change[len(r_change)-1] = beta_change[len(r_change)-1] + np.sum(np.abs(np.array(x_history[i][j][k+1])[n*7+6] - np.array(x_history[i][j][k])[n*7+6]))/(opt[j].N +1)

        # for l in range(len(r_change)):
        #     if r_change[l] == 0:
        #         r_change[l] = r_change[l-1]
        #         v_change[l] = v_change[l-1]
        #         beta_change[l] = beta_change[l-1]
        #     else:
        #         r_change[l] = np.log10(r_change[l])
        #         v_change[l] = np.log10(v_change[l])
        #         beta_change[l] = np.log10(beta_change[l])
        linetype = shape[j] + colors[i]

        ax1[i//2, i%2].plot(r_change, label=MHE_type[j])
        ax2[i//2, i%2].plot(v_change, label=MHE_type[j])
        ax3[i//2, i%2].plot(beta_change, label=MHE_type[j])

    ax1[i//2, i%2].set_yscale('log')
    ax2[i//2, i%2].set_yscale('log')
    ax3[i//2, i%2].set_yscale('log')
    ax1[i // 2, i % 2].set_title('Time (s) = ' + str(stop_points[i]))
    ax2[i // 2, i % 2].set_title('Time (s) = ' + str(stop_points[i]))
    ax3[i // 2, i % 2].set_title('Time (s) = ' + str(stop_points[i]))

for axs in ax1.flat:
    axs.set(xlabel='Iterations', ylabel=r'$||\Delta \hat r_i ||^2$')
for axs in ax1.flat:
    axs.label_outer()
for axs in ax2.flat:
    axs.set(xlabel='Iterations', ylabel=r'$||\Delta  \hat v_i ||^2$')
for axs in ax2.flat:
    axs.label_outer()
for axs in ax3.flat:
    axs.set(xlabel='Iterations', ylabel=r'$||\Delta \hat \beta_i ||^2$')
for axs in ax3.flat:
    axs.label_outer()

handles, labels = ax1[1, 0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='upper center', ncol=4)

handles, labels = ax2[1, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', ncol=4)

handles, labels = ax3[1, 0].get_legend_handles_labels()
fig3.legend(handles, labels, loc='upper center', ncol=4)
plt.show()

# for i in range(len(stop_points)):
#     plt.figure(i)
#     for j in range(len(opt)):
#         iter = np.linspace(0, 40, len(np.array(cost_history[i])[j]))
#         plt.plot(iter, np.array(cost_history[i])[j])
# plt.show()
