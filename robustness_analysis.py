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


t_lim = 130
N = [20]  # size of the horizon
measurement_lapse = 0.5  # time lapse between every measurement

t = 0.00
step = int(0)  # number of measurements measurements made
delta = int(0)
height = 80e3

# Initialisation of true dynamics and approximation model
o = SateliteObserver(22, 10)  #40.24, 3.42)
d = dynamics(height, 22, 0, 6000, -5, 60, o, wind='off', mass_change='off')
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
MHE_type = ['Multi-shooting']
method = ['Newton LS']
# measurement_pen =  [0.06, 75, 75] # coefficients obtained from the estimation opt of MS_MHE_PE
# model_pen =  [1e3, 1e3, 1e3, 1e1, 1e1, 1e1, 0.411]  # coefficients obtained from the estimation opt of MS_MHE_PE
measurement_pen =  [1e6, 1e1, 1e1]  # [1e7, 1, 1] #  [1e6, 1e-1, 1e-1] # [0.06, 80, 80] [1, 1e2, 1e3]  #
model_pen =  [1e-3,1e-3,1e-3, 5e-1,5e-1,5e-1, 1e-2] # [1e6, 1e6, 1e6, 1e1, 1e1, 1e1, 1e-1] #  [3, 3, 3, 1, 1, 1, 0.43] #[1, 1, 1, 1e1, 1e1, 1e1, 1e-1]
arrival = [1, 1]

for i in range(len(N)):
    if MHE_type[i] == 'Total ballistic':
        opt.append(total_ballistic(m, o, N[i], measurement_lapse, model_pen, method[i], Q))
    elif MHE_type[i] == 'Ballistic reg':
        opt.append(MHE_regularisation(m, o, N[i], measurement_lapse, model_pen, method[i], Q))
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

time = [0]
y_real = []
y_model = []
real_x = []
y_minus1 = o.h(d.r, 'on')

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
            y_real = [o.h(d.r, 'on')]
            m.reinitialise(y_minus1, y_real[0], o, measurement_lapse)

            y_model = [o.h(m.r)]
            real_x = [[d.r, d.v]]
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
# ---------------------------------------------------------------------------------------- #
print(y_covariance)
print('----')
print(process_covariance)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
sns.set()

# ig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(2,2)
# plt.subplot2grid((2,2), (0,0), colspan=2, rowspan=1)
theoretical_r = np.random.normal(0, Q[0,0]**0.5, size=10000)
theoretical_v = np.random.normal(0, Q[3,3]**0.5, size=10000)
theoretical_beta = np.random.normal(0, Q[6,6]**0.5, size=10000)

fig, axes = plt.subplots(1,3)
prob = pd.DataFrame(np.array(process_error)[:,0:3])
prob.plot(kind="kde", ax=axes[0])
prob = pd.DataFrame(theoretical_r)
prob.plot(kind="kde", style='--', color='k', ax=axes[0])
prob = pd.DataFrame(np.array(process_error)[:,3:6])
prob.plot(kind="kde", ax=axes[1])
prob = pd.DataFrame(theoretical_v)
prob.plot(kind="kde", style='--', color='k', ax=axes[1])
prob = pd.DataFrame(np.array(process_error)[:,6])
prob.plot(kind="kde", ax=axes[2])
prob = pd.DataFrame(theoretical_beta)
prob.plot(kind="kde", style='--', color='k', ax=axes[2])
axes[0].set_xlabel(r'$r \; (km)$')
axes[0].legend([r"$x$",r"$y$",r"$z$"]);
axes[1].set_xlabel(r'$v \; (m.s^{-1})$')
axes[1].legend([r"$x$",r"$y$",r"$z$"]);
axes[2].set_xlabel(r'$\beta \; (kg.m^{-2})$')
axes[2].legend([r"$\beta$"]);
plt.subplots_adjust(left=0.05, right=0.97, top=0.75, bottom=0.25)
plt.show()
