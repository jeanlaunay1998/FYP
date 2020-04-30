import matplotlib.pyplot as plt
import numpy as np

class Memory:
    def __init__(self, observer, N):
        self.N = N
        self.o = observer
        self.states = []
        self.y_estimates = []
        self.y_model = []
        self.cost = []
        self.t = []

    def save_data(self, time, vars, y_model, cost):
        ys = []
        xs = []
        for i in range(0,2*self.N+1,2):
            ys.append(self.o.h(vars[i*7:i*7+3], 'off'))
            xs.append(vars[i*7:(i+1)*7])

        self.states.append(xs)
        self.y_estimates.append(ys)
        self.y_model.append(y_model)
        self.cost.append(cost)
        self.t.append(time)

    def make_plots(self, real_x, real_beta, y_real, Sk):

        # ballistic coefficient plot
        plt.figure(1)
        plt.plot(self.t, real_beta[self.N:len(real_beta)], 'k', label='True system')
        plt.plot(self.t, np.array(self.states)[:, self.N, 6], '-+b', label='MHE')
        # plt.plot(self.t, beta_estimation1, '--g', label='MHE 2', markersize=5)
        # plt.plot(time, m.ballistic, 'r', label='Ballistic coef')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.ylabel('Ballistic Coefficient')
        plt.ylim((0,1000))

        # Measurements plot
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].plot(self.t, np.array(y_real)[self.N:len(y_real), 0], 'k')
        ax[0, 0].set(xlabel='Time (s)', ylabel='d (m)')
        ax[1, 0].plot(self.t, np.array(y_real)[self.N:len(y_real), 1], 'k')
        ax[1, 0].set(xlabel='Time (s)', ylabel='el (radians)')
        ax[2, 0].plot(self.t, np.array(y_real)[self.N:len(y_real), 2], 'k', label='True system')
        ax[2, 0].set(xlabel='Time (s)', ylabel='az (radians)')

        ax[0, 0].plot(self.t, np.array(self.y_model)[:, 0], 'r')
        ax[1, 0].plot(self.t, np.array(self.y_model)[:, 1], 'r')
        ax[2, 0].plot(self.t, np.array(self.y_model)[:, 2], 'r', label='Estimation model')

        ax[0, 0].plot(self.t, np.array(self.y_estimates)[:, self.N, 0], 'b')
        ax[1, 0].plot(self.t, np.array(self.y_estimates)[:, self.N, 1], 'b')
        ax[2, 0].plot(self.t, np.array(self.y_estimates)[:, self.N, 2], 'b', label='MHE')

        ax[0, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.y_model)[:, 0]-np.array(y_real)[self.N:len(y_real), 0], np.array(y_real)[self.N:len(y_real), 0],\
                                                   out=np.zeros_like(np.array(self.y_model)[:, 0]), where=np.array(y_real)[self.N:len(y_real), 0]!=0)), 'r')
        ax[0, 1].set(xlabel='Time (s)', ylabel='Error')
        ax[1, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.y_model)[:, 1]-np.array(y_real)[self.N:len(y_real), 1], np.array(y_real)[self.N:len(y_real), 1],\
                                                   out=np.zeros_like(np.array(self.y_model)[:, 1]), where=np.array(y_real)[self.N:len(y_real), 1]!=0)), 'r')
        ax[1, 1].set(xlabel='Time (s)', ylabel='Error')
        ax[2, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.y_model)[:, 2]-np.array(y_real)[self.N:len(y_real), 2], np.array(y_real)[self.N:len(y_real), 2],\
                                                   out=np.zeros_like(np.array(self.y_model)[:, 2]), where=np.array(y_real)[self.N:len(y_real), 2]!=0)), 'r', label='Estimation model')
        ax[2, 1].set(xlabel='Time (s)', ylabel='Error')

        ax[0, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.y_estimates)[:, self.N, 0]-np.array(y_real)[self.N:len(y_real), 0], np.array(y_real)[self.N:len(y_real), 0], \
                      out=np.zeros_like(np.array(self.y_estimates)[:, self.N, 0]), where=np.array(y_real)[self.N:len(y_real), 0]!=0)), 'b')
        ax[1, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.y_estimates)[:, self.N, 1]-np.array(y_real)[self.N:len(y_real), 1], np.array(y_real)[self.N:len(y_real), 1], \
                      out=np.zeros_like(np.array(self.y_estimates)[:, self.N, 1]), where=np.array(y_real)[self.N:len(y_real), 1]!=0)),'b')
        ax[2, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.y_estimates)[:, self.N, 2]-np.array(y_real)[self.N:len(y_real), 2], np.array(y_real)[self.N:len(y_real), 2], \
                      out=np.zeros_like(np.array(self.y_estimates)[:, self.N, 2]), where=np.array(y_real)[self.N:len(y_real), 2]!=0)), 'b', label='MHE')
        plt.legend(loc='best')

        # position plot
        fig, ax = plt.subplots(3, 2)
        real = np.array(real_x)
        ax[0, 0].plot(self.t, real[self.N:len(Sk), 0, 0], 'k')
        ax[0, 0].set(xlabel='Time (s)', ylabel='Position x (m)')
        ax[1, 0].plot(self.t, real[self.N:len(y_real), 0, 1], 'k')
        ax[1, 0].set(xlabel='Time (s)', ylabel='Position y (m)')
        ax[2, 0].plot(self.t, real[self.N:len(y_real), 0, 2], 'k', label='True system')
        ax[2, 0].set(xlabel='Time (s)', ylabel='Position z (m)')

        ax[0, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 0, 0], 'r')
        ax[1, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 0, 1], 'r')
        ax[2, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 0, 2], 'r', label='Estimation Model')

        ax[0, 0].plot(self.t, np.array(self.states)[:, self.N, 0], 'b')
        ax[1, 0].plot(self.t, np.array(self.states)[:, self.N, 1], 'b')
        ax[2, 0].plot(self.t, np.array(self.states)[:, self.N, 2], 'b', label='MHE')

        ax[0, 1].plot(self.t, 100 * np.abs(np.divide(np.array(self.states)[:, self.N, 0] - real[self.N:len(Sk), 0, 0],  real[self.N:len(Sk), 0, 0], \
                                                     out=np.zeros_like(real[self.N:len(Sk), 0, 0]), where=real[self.N:len(Sk), 0, 0]!=0)), 'b')
        ax[0, 1].set(xlabel='Time (s)', ylabel='Error')
        ax[1, 1].plot(self.t, 100 * np.abs(np.divide(np.array(self.states)[:, self.N, 1] - real[self.N:len(Sk), 0, 1],  real[self.N:len(Sk), 0, 1], \
                                                     out=np.zeros_like(real[self.N:len(Sk), 0, 1]), where=real[self.N:len(Sk), 0, 1]!=0)), 'b')
        ax[1, 1].set(xlabel='Time (s)', ylabel='Error')
        ax[2, 1].plot(self.t, 100 * np.abs(np.divide(np.array(self.states)[:, self.N, 2] - real[self.N:len(Sk), 0, 2],  real[self.N:len(Sk), 0, 2], \
                                                     out=np.zeros_like(real[self.N:len(Sk), 0, 2]), where=real[self.N:len(Sk), 0, 2]!=0)), 'b', label='MHE')
        ax[2, 1].set(xlabel='Time (s)', ylabel='Error')

        ax[0, 1].plot(self.t, 100 * np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 0, 0] - real[self.N:len(Sk), 0, 0], real[self.N:len(Sk), 0, 0], \
                                                     out=np.zeros_like(real[self.N:len(Sk), 0, 0]), where=real[self.N:len(Sk), 0, 0]!=0)), 'r')
        ax[1, 1].plot(self.t, 100 * np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 0, 1] - real[self.N:len(Sk), 0, 1], real[self.N:len(Sk), 0, 1], \
                                                     out=np.zeros_like(real[self.N:len(Sk), 0, 1]), where=real[self.N:len(Sk), 0, 1]!=0)), 'r')
        ax[2, 1].plot(self.t, 100 * np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 0, 2] - real[self.N:len(Sk), 0, 2], real[self.N:len(Sk), 0, 2], \
                                                     out=np.zeros_like(real[self.N:len(Sk), 0, 2]), where=real[self.N:len(Sk), 0, 2]!=0)), 'r', label='Model')
        plt.legend(loc='best')


        fig, ax = plt.subplots(3, 2)
        real = np.array(real_x)
        ax[0, 0].plot(self.t, real[self.N:len(Sk), 1, 0], 'k')
        ax[0, 0].set(xlabel='Time (s)', ylabel='Velocity x (m/s)')
        ax[1, 0].plot(self.t, real[self.N:len(y_real), 1, 1], 'k')
        ax[1, 0].set(xlabel='Time (s)', ylabel='Velocity y (m/s)')
        ax[2, 0].plot(self.t, real[self.N:len(y_real), 1, 2], 'k', label='True system')
        ax[2, 0].set(xlabel='Time (s)', ylabel='Velocity z (m/s)')

        ax[0, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 1, 0], 'r')
        ax[1, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 1, 1], 'r')
        ax[2, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 1, 2], 'r', label='Estimation Model')

        ax[0, 0].plot(self.t, np.array(self.states)[:, self.N, 3], 'b')
        ax[1, 0].plot(self.t, np.array(self.states)[:, self.N, 4], 'b')
        ax[2, 0].plot(self.t, np.array(self.states)[:, self.N, 5], 'b', label='MHE')

        ax[0, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.states)[:, self.N, 3]-real[self.N:len(Sk), 1, 0], real[self.N:len(Sk), 1, 0],\
                                                   out=np.zeros_like(np.array(self.states)[:, self.N, 3]), where=real[self.N:len(Sk), 1, 0]!=0)), 'b')
        ax[0, 1].set(xlabel='Time (s)', ylabel='Error')
        ax[1, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.states)[:, self.N, 4]-real[self.N:len(Sk), 1, 1], real[self.N:len(Sk), 1, 1],\
                                                   out=np.zeros_like(np.array(self.states)[:, self.N, 4]), where=real[self.N:len(Sk), 1, 1]!=0)), 'b')
        ax[1, 1].set(xlabel='Time (s)', ylabel='Error')
        ax[2, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.states)[:, self.N, 5]-real[self.N:len(Sk), 1, 2], real[self.N:len(Sk), 1, 2],\
                                                   out=np.zeros_like(np.array(self.states)[:, self.N, 5]), where=real[self.N:len(Sk), 1, 2]!=0)), 'b', label='MHE')
        ax[2, 1].set(xlabel='Time (s)', ylabel='Error')

        ax[0, 1].plot(self.t, 100*np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 1, 0]-real[self.N:len(Sk), 1, 0], real[self.N:len(Sk), 1, 0],\
                                                   out=np.zeros_like(np.array(Sk)[self.N:len(y_real), 1, 0]), where=real[self.N:len(Sk), 1, 0]!=0)), 'r')
        ax[1, 1].plot(self.t, 100*np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 1, 1]-real[self.N:len(Sk), 1, 1], real[self.N:len(Sk), 1, 1],\
                                                   out=np.zeros_like(np.array(Sk)[self.N:len(y_real), 1, 1]), where=real[self.N:len(Sk), 1, 1]!=0)), 'r')
        ax[2, 1].plot(self.t, 100*np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 1, 2]-real[self.N:len(Sk), 1, 2], real[self.N:len(Sk), 1, 2],\
                                                   out=np.zeros_like(np.array(Sk)[self.N:len(y_real), 1, 2]), where=real[self.N:len(Sk), 1, 2]!=0)), 'r', label='Model')

        plt.legend(loc='best')
        plt.show()




