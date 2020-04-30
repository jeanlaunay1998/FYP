import numpy.linalg as LA
import numpy as np
import sys

from system_dynamics import dynamics
from observer import SateliteObserver
from state_model import model
from multi_shooting_MHE import multishooting
from memory import Memory
t_lim = 150
N = 50  # size of the horizon
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
opt = multishooting(m, d, o, N, measurement_lapse)
memory = Memory(o, N)

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
        else:
            m.step_update()  # the model is updated every 0.5 seconds (problem with discretization)
            y_real.append(o.h(d.r))

            # re-initialise model from taken measurements
            y_model.append(o.h(m.r, 'off'))
            time.append(t)
            real_x.append([d.r, d.v])
            real_beta.append(d.beta[len(d.beta) - 1])

        if step >= opt.N+1: # MHE is entered only when there exists sufficient measurements over the horizon
            if step==opt.N+1:
                opt.estimator_initilisation(step, y_real)
            else:
                opt.slide_window(y_real[step-1])

            opt.estimation()
            memory.save_data(t, opt.vars, o.h(m.r, 'off'), opt.cost(opt.vars))

memory.make_plots(real_x, real_beta, y_real, m.Sk)


