import matplotlib.pyplot as plt
import numpy as np
import sys

class Memory:
    def __init__(self, observer, N, estimator_number):
        self.N = N
        self.o = observer
        self.y_model = []
        self.t = []
        self.states = []
        self.y_estimates = []
        self.cost = []
        for i in range(estimator_number):
            self.states.append([])
            self.y_estimates.append([])
            self.cost.append([])
        self.size = estimator_number

    def save_data(self, time, vars, y_model, cost, number):
        ys = []
        xs = []
        # for i in range(0,self.N[number]+1):
        for i in range(0,2*self.N[number]+1,2):
            ys.append(self.o.h(vars[i*7:i*7+3], 'off'))
            xs.append(vars[i*7:(i+1)*7])

        self.states[number].append(np.copy(xs))
        self.y_estimates[number].append(ys)
        self.cost[number].append(cost)
        if number == 0:
            self.y_model.append(y_model)
            self.t.append(time)

    def make_plots(self, real_x, real_beta, y_real, Sk, KF_states, EKF_states):
        kf_y = []
        ekf_y = []
        for i in range(len(KF_states)):
            kf_y.append(self.o.h(KF_states[i]))
            ekf_y.append(self.o.h(EKF_states[i]))

        labelstring = []
        method = ['Newton LS', 'Newton']
        for i in range(self.size):
            labelstring.append('MHE N =' + str(self.N[i]))
            # labelstring.append('MHE with ' + method[i])

            # ballistic coefficient plot
        plt.figure(1)
        plt.plot(self.t, real_beta[self.N[0]:len(real_beta)], 'k', label='True system')
        plt.plot(self.t, np.array(KF_states)[self.N[0]:len(KF_states), 6], 'b', label='Unscented Kalman filter')
        plt.plot(self.t, np.array(EKF_states)[self.N[0]:len(EKF_states), 6], 'r', label='Extended Kalman filter')
        for i in range(self.size):
            plt.plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.states[i])[:, self.N[i], 6], '-', label=labelstring[i])
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.ylabel('Ballistic Coefficient')

        # Measurements plot
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].plot(self.t, np.array(y_real)[self.N[0]:len(y_real), 0], 'k')
        ax[0, 0].plot(self.t, np.array(kf_y)[self.N[0]:len(KF_states), 0], 'b')
        ax[0, 0].plot(self.t, np.array(ekf_y)[self.N[0]:len(EKF_states), 0], 'r')
        ax[0, 0].set(xlabel='Time (s)', ylabel='d (m)')
        ax[1, 0].plot(self.t, np.array(y_real)[self.N[0]:len(y_real), 1], 'k')
        ax[1, 0].plot(self.t, np.array(kf_y)[self.N[0]:len(KF_states), 1], 'b')
        ax[1, 0].plot(self.t, np.array(ekf_y)[self.N[0]:len(EKF_states), 1], 'r')
        ax[1, 0].set(xlabel='Time (s)', ylabel='el (radians)')
        ax[2, 0].plot(self.t, np.array(y_real)[self.N[0]:len(y_real), 2], 'k', label='True system')
        ax[2, 0].plot(self.t, np.array(kf_y)[self.N[0]:len(KF_states), 2], 'b', label='Unscented Kalman filter')
        ax[2, 0].plot(self.t, np.array(ekf_y)[self.N[0]:len(EKF_states), 2], 'b', label='Extended Kalman filter')
        ax[2, 0].set(xlabel='Time (s)', ylabel='az (radians)')

        # plot Estimates

        for j in range(3):
                y = 100*np.abs(np.divide(np.array(kf_y)[self.N[0]:len(KF_states), j] - np.array(y_real)[self.N[0]:len(y_real), j], np.array(y_real)[self.N[0]:len(y_real), j], \
                                         out=np.array(y_real)[self.N[0]:len(y_real), j], where=np.array(y_real)[self.N[0]:len(y_real), j]!=0))
                y1 = 100*np.abs(np.divide(np.array(ekf_y)[self.N[0]:len(EKF_states), j] - np.array(y_real)[self.N[0]:len(y_real), j], np.array(y_real)[self.N[0]:len(y_real), j], \
                                         out=np.array(y_real)[self.N[0]:len(y_real), j], where=np.array(y_real)[self.N[0]:len(y_real), j]!=0))
                ax[j, 1].plot(self.t, y, '-b')
                ax[j, 1].plot(self.t, y1, '-b')

        lim = [0.5, 5, 30]
        for i in range(self.size):
            ax[0, 0].plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.y_estimates[i])[:, self.N[i], 0], '-')
            ax[1, 0].plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.y_estimates[i])[:, self.N[i], 1], '-')
            ax[2, 0].plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.y_estimates[i])[:, self.N[i], 2], '-', label=labelstring[i])
            for j in range(3):
                y = 100*np.abs(np.divide(np.array(self.y_estimates[i])[:, self.N[i], j]-np.array(y_real)[self.N[i]:len(y_real), j], np.array(y_real)[self.N[i]:len(y_real), j], \
                                         out=np.array(y_real)[self.N[i]:len(y_real), j], where=np.array(y_real)[self.N[i]:len(y_real), j]!=0))
                ax[j, 1].plot(self.t[self.N[i]-self.N[0]:len(self.t)], y, '-')

                ax[j, 0].set_ylim([np.amin(np.array(y_real)[self.N[0]:len(y_real), j]) - 0.1*np.abs(np.amin(np.array(y_real)[self.N[0]:len(y_real), j])), \
                                   np.amax(np.array(y_real)[self.N[0]:len(y_real), j]) + 0.1*np.abs(np.amax(np.array(y_real)[self.N[0]:len(y_real), j]))])
                ax[j, 1].set_ylim([0, lim[j]])
                ax[j, 1].set(xlabel='Time (s)', ylabel='Error')

        handles, labels = ax[2, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)


        # position plot
        fig, ax = plt.subplots(3, 2)
        real = np.array(real_x)
        ax[0, 0].plot(self.t, real[self.N[0]:len(y_real), 0, 0], 'k')
        ax[0, 0].plot(self.t, np.array(KF_states)[self.N[0]:len(KF_states), 0], 'b')
        ax[0, 0].plot(self.t, np.array(EKF_states)[self.N[0]:len(EKF_states), 0], 'r')
        ax[0, 0].set(xlabel='Time (s)', ylabel='Position x (m)')
        ax[1, 0].plot(self.t, real[self.N[0]:len(y_real), 0, 1], 'k')
        ax[1, 0].plot(self.t, np.array(KF_states)[self.N[0]:len(KF_states), 1], 'b')
        ax[1, 0].plot(self.t, np.array(EKF_states)[self.N[0]:len(EKF_states), 1], 'r')
        ax[1, 0].set(xlabel='Time (s)', ylabel='Position y (m)')
        ax[2, 0].plot(self.t, real[self.N[0]:len(y_real), 0, 2], 'k', label='True system')
        ax[2, 0].plot(self.t, np.array(KF_states)[self.N[0]:len(KF_states), 2], 'b', label='Unscented Kalman filter')
        ax[2, 0].plot(self.t, np.array(EKF_states)[self.N[0]:len(EKF_states), 2], 'r', label='Extended Kalman filter')
        ax[2, 0].set(xlabel='Time (s)', ylabel='Position z (m)')
        lim = [1, 5, 1]

        for j in range(3):
            y = np.abs(np.divide(np.array(KF_states)[self.N[0]:len(KF_states), j] - real[self.N[0]:len(Sk), 0, j],  real[self.N[0]:len(Sk), 0, j], \
                                     out=np.zeros_like(real[self.N[i]:len(Sk), 0, j]), where=real[self.N[i]:len(Sk), 0, j]!=0))
            ax[j, 1].plot(self.t, 100 * y, '-b')
            y = np.abs(np.divide(np.array(EKF_states)[self.N[0]:len(EKF_states), j] - real[self.N[0]:len(Sk), 0, j],  real[self.N[0]:len(Sk), 0, j], \
                                     out=np.zeros_like(real[self.N[i]:len(Sk), 0, j]), where=real[self.N[i]:len(Sk), 0, j]!=0))
            ax[j, 1].plot(self.t, 100 * y, '-r')

        for i in range(self.size):
            ax[0, 0].plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.states[i])[:, self.N[i], 0], '-')
            ax[1, 0].plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.states[i])[:, self.N[i], 1], '-')
            ax[2, 0].plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.states[i])[:, self.N[i], 2], '-', label=labelstring[i])
            for j in range(3):
                y = np.abs(np.divide(np.array(self.states[i])[:, self.N[i], j] - real[self.N[i]:len(Sk), 0, j],  real[self.N[i]:len(Sk), 0, j], \
                                     out=np.zeros_like(real[self.N[i]:len(Sk), 0, j]), where=real[self.N[i]:len(Sk), 0, j]!=0))
                ax[j, 1].plot(self.t[self.N[i]-self.N[0]:len(self.t)], 100 * y, '-')
                ax[j, 1].set_ylim([0, lim[i]])
                ax[j, 0].set_ylim([np.amin(real[self.N[0]:len(Sk), 0, j]) - 0.1*np.abs(np.amin(real[self.N[0]:len(Sk), 0, j])), \
                                   np.amax(real[self.N[0]:len(Sk), 0, j]) + 0.1*np.abs(np.amax(real[self.N[0]:len(Sk), 0, j]))])
                ax[j, 1].set(xlabel='Time (s)', ylabel='Error')
        handles, labels = ax[2,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',  ncol=4)



        fig, ax = plt.subplots(3, 2)

        ax[0, 0].plot(self.t, real[self.N[0]:len(y_real), 1, 0], 'k')
        ax[0, 0].plot(self.t, np.array(KF_states)[self.N[0]:len(KF_states), 3], 'b')
        ax[0, 0].plot(self.t, np.array(EKF_states)[self.N[0]:len(EKF_states), 3], 'r')
        ax[0, 0].set(xlabel='Time (s)', ylabel='Velocity x (m/s)')
        ax[1, 0].plot(self.t, real[self.N[0]:len(y_real), 1, 1], 'k')
        ax[1, 0].plot(self.t, np.array(KF_states)[self.N[0]:len(KF_states), 4], 'b')
        ax[1, 0].plot(self.t, np.array(EKF_states)[self.N[0]:len(EKF_states), 4], 'r')
        ax[1, 0].set(xlabel='Time (s)', ylabel='Velocity y (m/s)')
        ax[2, 0].plot(self.t, real[self.N[0]:len(y_real), 1, 2], 'k', label='True system')
        ax[2, 0].plot(self.t, np.array(KF_states)[self.N[0]:len(KF_states), 5], 'b', label='Unscented Kalman filter')
        ax[2, 0].plot(self.t, np.array(EKF_states)[self.N[0]:len(EKF_states), 5], 'r', label='Extended Kalman filter')
        ax[2, 0].set(xlabel='Time (s)', ylabel='Velocity z (m/s)')


        for j in range(3):
            y = np.abs(np.divide(np.array(KF_states)[self.N[0]:len(KF_states), j+3] - real[self.N[i]:len(Sk), 1, j],  real[self.N[i]:len(Sk), 1, j], \
                                     out=np.zeros_like(real[self.N[i]:len(Sk), 1, j]), where=real[self.N[i]:len(Sk), 1, j]!=0))
            ax[j, 1].plot(self.t, 100 * y, '-b')

        for i in range(self.size):
            ax[0, 0].plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.states[i])[:, self.N[i], 3], '-')
            ax[1, 0].plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.states[i])[:, self.N[i], 4], '-')
            ax[2, 0].plot(self.t[self.N[i]-self.N[0]:len(self.t)], np.array(self.states[i])[:, self.N[i], 5], '-', label=labelstring[i])
            for j in range(3):
                y = np.abs(np.divide(np.array(self.states[i])[:, self.N[i], j+3] - real[self.N[i]:len(Sk), 1, j],  real[self.N[i]:len(Sk), 1, j], \
                                     out=np.zeros_like(real[self.N[i]:len(Sk), 1, j]), where=real[self.N[i]:len(Sk), 1, j]!=0))
                ax[j, 1].plot(self.t[self.N[i]-self.N[0]:len(self.t)], 100 * y, '-')

                ax[j, 1].set_ylim([0, 30])
                ax[j, 0].set_ylim([np.amin(real[self.N[0]:len(Sk), 1, j]) - 0.1*np.abs(np.amin(real[self.N[0]:len(Sk), 1, j])), \
                                   np.amax(real[self.N[0]:len(Sk), 1, j]) + 0.1*np.abs(np.amax(real[self.N[0]:len(Sk), 1, j]))])
                ax[j, 1].set(xlabel='Time (s)', ylabel='Error')

        handles, labels = ax[2,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',  ncol=4)


        plt.show()

# Plot approximation model variables
        # ax[0, 0].plot(self.t, np.array(self.y_model)[:, 0], 'r')
        # ax[1, 0].plot(self.t, np.array(self.y_model)[:, 1], 'r')
        # ax[2, 0].plot(self.t, np.array(self.y_model)[:, 2], 'r', label='Estimation model')
        #
        # ax[0, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.y_model)[:, 0]-np.array(y_real)[self.N:len(y_real), 0], np.array(y_real)[self.N:len(y_real), 0],\
        #                                            out=np.zeros_like(np.array(self.y_model)[:, 0]), where=np.array(y_real)[self.N:len(y_real), 0]!=0)), 'r')
        # ax[1, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.y_model)[:, 1]-np.array(y_real)[self.N:len(y_real), 1], np.array(y_real)[self.N:len(y_real), 1],\
        #                                            out=np.zeros_like(np.array(self.y_model)[:, 1]), where=np.array(y_real)[self.N:len(y_real), 1]!=0)), 'r')
        # ax[2, 1].plot(self.t, 100*np.abs(np.divide(np.array(self.y_model)[:, 2]-np.array(y_real)[self.N:len(y_real), 2], np.array(y_real)[self.N:len(y_real), 2],\
        #                                            out=np.zeros_like(np.array(self.y_model)[:, 2]), where=np.array(y_real)[self.N:len(y_real), 2]!=0)), 'r', label='Estimation model')



# approximation model plots
        # ax[0, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 0, 0], 'r')
        # ax[1, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 0, 1], 'r')
        # ax[2, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 0, 2], 'r', label='Estimation Model')
        # ax[0, 1].plot(self.t, 100 * np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 0, 0] - real[self.N:len(Sk), 0, 0], real[self.N:len(Sk), 0, 0], \
        #                                              out=np.zeros_like(real[self.N:len(Sk), 0, 0]), where=real[self.N:len(Sk), 0, 0]!=0)), 'r')
        # ax[1, 1].plot(self.t, 100 * np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 0, 1] - real[self.N:len(Sk), 0, 1], real[self.N:len(Sk), 0, 1], \
        #                                              out=np.zeros_like(real[self.N:len(Sk), 0, 1]), where=real[self.N:len(Sk), 0, 1]!=0)), 'r')
        # ax[2, 1].plot(self.t, 100 * np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 0, 2] - real[self.N:len(Sk), 0, 2], real[self.N:len(Sk), 0, 2], \
        #                                              out=np.zeros_like(real[self.N:len(Sk), 0, 2]), where=real[self.N:len(Sk), 0, 2]!=0)), 'r', label='Model')


 # ax[0, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 1, 0], 'r')
        # ax[1, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 1, 1], 'r')
        # ax[2, 0].plot(self.t, np.array(Sk)[self.N:len(y_real), 1, 2], 'r', label='Estimation Model')
        # ax[0, 1].plot(self.t, 100*np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 1, 0]-real[self.N:len(Sk), 1, 0], real[self.N:len(Sk), 1, 0],\
        #                                            out=np.zeros_like(np.array(Sk)[self.N:len(y_real), 1, 0]), where=real[self.N:len(Sk), 1, 0]!=0)), 'r')
        # ax[1, 1].plot(self.t, 100*np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 1, 1]-real[self.N:len(Sk), 1, 1], real[self.N:len(Sk), 1, 1],\
        #                                            out=np.zeros_like(np.array(Sk)[self.N:len(y_real), 1, 1]), where=real[self.N:len(Sk), 1, 1]!=0)), 'r')
        # ax[2, 1].plot(self.t, 100*np.abs(np.divide(np.array(Sk)[self.N:len(y_real), 1, 2]-real[self.N:len(Sk), 1, 2], real[self.N:len(Sk), 1, 2],\
        #                                            out=np.zeros_like(np.array(Sk)[self.N:len(y_real), 1, 2]), where=real[self.N:len(Sk), 1, 2]!=0)), 'r', label='Model')
