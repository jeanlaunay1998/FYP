import numpy.linalg as LA
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
import sys

from system_dynamics import dynamics
from observer import SateliteObserver
from state_model import model
from multi_shooting_MHE import multishooting
from memory import Memory

t_lim = 150
N = [20]  # size of the horizon
measurement_lapse = 0.5  # time lapse between every measurement

t = 0.00
step = int(0)  # number of measurements measurements made
delta = int(0)
# import pdb; pdb.set_trace()

height = 80e3
d = dynamics(height, 22, 0, 6000, -5, 60)
o = SateliteObserver(40.24, 3.42)
initialbeta = d.beta[0] + np.random.normal(0, 0.01*d.beta[0], size=1)[0]
m = model(d.r, d.v, initialbeta, measurement_lapse)

opt = []
method = ['Newton LS']
measurement_pen = [1e6, 1e-1, 1e-1]
model_pen = [1e4, 1e4, 1e4, 1e1, 1e1, 1e1, 1e0]
for i in range(len(N)):
    opt.append(multishooting(m, d, o, N[i], measurement_lapse, measurement_pen, model_pen, method[i]))
memory = Memory(o, N, len(N))

# covariance matrices
R = np.array([[50**2, 0, 0], [0, (1e-3)**2, 0], [0, 0, (1e-3)**2]]) # Measurement covariance matrix
P0 = np.zeros((7,7))  # Initial covariance matrix
Q = np.zeros((7,7))  # Process noise covariance matrix
qa = 10 #  Estimated deviation of acceleration between real state and approximated state

for i in range(3):
    P0[i,i] = 500**2
    P0[i+3, i+3] = 1000**2

    Q[i, i] = (qa*measurement_lapse**3)/3
    Q[i, i+3] = (qa*measurement_lapse**2)/2
    Q[i+3, i] = (qa*measurement_lapse**2)/2
    Q[i+3, i+3] = qa*measurement_lapse
P0[6,6] = 20**2
Q[6,6] = 100
points = MerweScaledSigmaPoints(n=7, alpha=.1, beta=2., kappa=-1)
ukf = UKF(dim_x=7, dim_z=3, fx=m.f, hx=o.h, dt=measurement_lapse, points=points)
ukf.P = P0
ukf.Q = Q
ukf.R = R


time = [0]
y_real = []
y_model = []
real_x = []
y_minus1 = o.h(d.r, 'off')
penalties = []

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

            # initialisation of the unscented kalman filter
            ukf.x = np.array([m.r[0], m.r[1], m.r[2], m.v[0], m.v[1], m.v[2], m.beta])
            UKF_state = [np.copy(ukf.x)]

        else:
            m.step_update()  # the model is updated every 0.5 seconds (problem with discretization)
            y_real.append(o.h(d.r))

            # re-initialise model from taken measurements
            y_model.append(o.h(m.r, 'off'))
            time.append(t)
            real_x.append([d.r, d.v])
            real_beta.append(d.beta[len(d.beta) - 1])

            ukf.predict()
            ukf.update(y_real[len(y_real)-1])
            UKF_state.append(np.copy(ukf.x))

        for i in range(len(opt)):
            if step >= opt[i].N+1: # MHE is entered only when there exists sufficient measurements over the horizon
                if step==opt[i].N+1:
                    opt[i].estimator_initilisation(step, y_real)
                    opt[i].tuning_MHE(real_x, real_beta, step)
                else:
                    opt[i].slide_window(y_real[step-1])
                    opt[i].tuning_MHE(real_x, real_beta, step)
                penalties.append([opt[i].measurement_pen[0],opt[i].measurement_pen[1],opt[i].measurement_pen[2],\
                                 opt[i].model_pen[0], opt[i].model_pen[1],opt[i].model_pen[2],opt[i].model_pen[3],opt[i].model_pen[4],opt[i].model_pen[5],opt[i].model_pen[6]])
                opt[i].estimation()
                memory.save_data(t, opt[i].vars, o.h(m.r, 'off'), opt[i].cost(opt[i].vars), i)

memory.make_plots(real_x, real_beta, y_real, m.Sk, UKF_state)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(5, 2)
x = np.linspace(0, 15, len(penalties))
for i in range(10):
    ax[i%5, i//5].plot(np.array(penalties)[:][i])
plt.show()

# ----------------------------------------------------------------------------------------------------- #




