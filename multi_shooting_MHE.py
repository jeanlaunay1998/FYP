from numpy import linalg as LA
import numpy as np
from newton_method import newton_iter_selection
from newton_method import BFGS
from newton_method import gradient_search
from newton_method import newton_iter
import sys
from scipy.optimize import minimize

class multishooting:
    def __init__(self, estimation_model, true_system, observer, horizon, measurement_lapse, pen1, pen2, Q, R, opt_method='Newton LS'):
        self.N = horizon
        self.m = estimation_model
        self.d = true_system
        self.o = observer
        self.method = opt_method
        self.Q = Q
        self.R = R

        self.inter_steps = int(measurement_lapse / self.m.delta_t)

        # the state x = {r, v, beta}
        self.vars = np.ones((1+2*self.N)*7) # list of inputs to the optimization
        self.y = []  # list of true measurements across the horizon
        self.pen = 1  # penalty factor
        self.pen1 = 0  # factor to remove Lagrangians (note this is highly inefficient since there will be un-used variables)

        # self.reg1 = self.R # LA.inv(self.R)  # np.zeros(3)  # distance, azimuth, elevation
        # self.reg2 = self.Q # LA.inv(self.Q)  # np.zeros(7)  # position, velocity and ballistic coeff

        self.reg1 = np.identity(3)
        self.reg2 = np.identity(7)*1e-5
        self.reg2[6,6] = 0.1*self.reg2[6,6]

        self.measurement_pen = pen1
        self.model_pen = pen2

        for i in range(3):
            self.reg1[i,i] = self.measurement_pen[i]
        for i in range(7):
            self.reg2[i,i] = self.model_pen[i]

        # VARIABLES FOR ARRIVAL COST
        self.x_prior = np.zeros(7)
        self.mu = 1e0


    def estimator_initilisation(self, step, y_measured):
        # this function initiates the estimator
        self.y = np.array(y_measured)[step - self.N - 1:step, :]
        for i in range(0, 2*self.N+1, 2):
            for j in range(3):
                self.vars[i*7+j] = np.copy(self.m.Sk[len(self.m.Sk)-(1+self.N)*self.inter_steps + i//2][0][j])
            for j in range(3,6):
                self.vars[i * 7 + j] = np.copy(self.m.Sk[len(self.m.Sk) - (1 + self.N) * self.inter_steps + i//2][1][j-3])
            self.vars[i*7 + 6] = self.m.beta
        self.x_prior = self.vars[0:7]

        # self.reg1 = np.ones(3)
        # self.reg2 = np.ones(7)
        #
        # self.reg1 = np.multiply(self.reg1, self.measurement_pen)
        # self.reg2 = np.multiply(self.reg2, self.model_pen)
        # it is assumed that the horizon is sufficiently small such that all measurements are of the same order at
        # end and beginning of the horizon
        # measurements reg

        for i in range(3):
            if np.abs(self.y[0, i]) < 1:
                mult = 1
                while mult * np.abs(self.y[0, i]) <= 1:
                    mult = mult * 10
                self.reg1[i, i] = self.measurement_pen[i] * mult
            else:
                mult = 1
                while np.abs(self.y[0, i]) // mult >= 10:
                    mult = mult * 10
                self.reg1[i, i] = self.measurement_pen[i] / mult

        # position and velocity reg
        for i in range(2):
            mult = 1
            if np.abs(self.vars[i * 3]) < 1:
                while mult * np.abs(self.vars[i * 3]) <= 1:
                    mult = mult * 10
                for j in range(3): self.reg2[i * 3 + j, i * 3 + j] = self.model_pen[i * 3 + j] * mult
            else:
                while np.abs(self.vars[i * 3]) // mult >= 10:
                    mult = mult * 10
                for j in range(3): self.reg2[i * 3 + j, i * 3 + j] = self.model_pen[i * 3 + j] / mult

        # ballistic coeff reg
        mult = 1
        while np.abs(self.vars[6]) // mult >= 10:
            mult = mult * 10
        self.reg2[6, 6] = self.model_pen[6] / mult



    def cost(self, x):
        if len(x) != len(self.vars):
            var = np.ones((2*self.N+1)*7)
            for i in range(0,2*self.N+1,2):
                var[i*7:(i+1)*7] = x[7*i//2:(1+i//2)*7]
        else:
            var = np.copy(x)
        h_i = []
        f_i = np.zeros((self.N+1)*7)

        for i in range(0, 2*self.N+1, 2):
            h_i.append(self.o.h(var[i*7:i*7+3], 'off'))
            f_i[(i//2)*7:(i//2)*7+3], f_i[(i//2)*7+3:(i//2)*7+6], a, f_i[(i//2)*7+6] = self.m.f(var[i*7:i*7+3], var[i*7+3:i*7+6], var[i*7+6], 'off')

        R_mu = np.identity(7)
        for i in range(7): R_mu[i, i] = self.reg2[i, i]/self.model_pen[i]

        # Arrival cost
        J = 0.5*self.mu*LA.norm(np.matmul(R_mu, self.vars[0:7] - self.x_prior))**2

        for i in range(self.N + 1):
            J = J + 0.5 * LA.norm(np.matmul(self.reg1, self.y[i] - h_i[i]))**2

        for i in range(0, 2*self.N, 2):
            J = J + self.pen1*np.matmul(var[(i+1)*7:(i+2)*7], np.matmul(self.reg2, var[(i+2)*7:(i+3)*7] - f_i[(i//2)*7:(i//2+1)*7])) \
                + 0.5*self.pen*LA.norm(np.matmul(self.reg2, var[(i+2)*7:(i+3)*7] - f_i[(i//2)*7:(i//2+1)*7]))**2
        return J


    def density_constants(self, height):
        height = height - self.m.R
        if height < 9144:
            c1 = 1.227
            c2 = 1.093e-4
        else:
            c1 = 1.754
            c2 = 1.490e-4
        return [c1, c2]


    def dacc_dr(self, r, v, beta):
        norm_r = LA.norm(r)
        norm_v = LA.norm(v)

        # For legibility of the gradient constant terms across gradient are previously defined
        constant1 = self.m.G*self.m.M*pow(norm_r, -3)
        c1, c2 = self.density_constants(norm_r)
        constant2 = norm_v*c2*self.m.density_h(r)/(2*beta*norm_r)


        dA = constant1 * np.array([[-1 + pow(norm_r, -2)*r[0]*r[0], pow(norm_r, -2)*r[0]*r[1], pow(norm_r, -2)*r[0]*r[2]],
                                   [pow(norm_r, -2)*r[1]*r[0], -1 + pow(norm_r, -2)*r[1]*r[1], pow(norm_r, -2)*r[1]*r[2]],
                                   [pow(norm_r, -2)*r[2]*r[0], pow(norm_r, -2)*r[2]*r[1], -1 + pow(norm_r, -2)*r[2]*r[2]]])

        dB = np.array([[constant2*v[0]*r[0], constant2*v[0]*r[1], constant2*v[0]*r[2]],
                       [constant2*v[1]*r[0], constant2*v[1]*r[1], constant2*v[1]*r[2]],
                       [constant2*v[2]*r[0], constant2*v[2]*r[1], constant2*v[2]*r[2]]])

        dadr = dA + dB
        return dadr


    def dacc_dv(self, r, v, beta):
        norm_v = LA.norm(v)
        constant1 = -(self.m.density_h(r)/(2*beta))
        dadv = np.array([[norm_v + pow(norm_v, -1)*v[0]*v[0], pow(norm_v, -1)*v[0]*v[1], pow(norm_v, -1)*v[0]*v[2]],
                         [pow(norm_v, -1)*v[1]*v[0], norm_v + pow(norm_v, -1)*v[1]*v[1], pow(norm_v, -1)*v[1]*v[2]],
                         [pow(norm_v, -1)*v[2]*v[0], pow(norm_v, -1)*v[2]*v[1], norm_v + pow(norm_v, -1)*v[2]*v[2]]])
        dadv = constant1*dadv
        return dadv


    def dacc_dbeta(self, r, v, beta):
        v_norm = LA.norm(v)

        da_dbeta = [0, 0, 0]
        da_dbeta[0] = 0.5*self.m.density_h(r)*v_norm*v[0]/(beta**2)
        da_dbeta[1] = 0.5*self.m.density_h(r)*v_norm*v[1]/(beta**2)
        da_dbeta[2] = 0.5*self.m.density_h(r)*v_norm*v[2]/(beta**2)

        return da_dbeta


    def dfdx(self, x):
        # x: point at which the derivative is evaluated
        r = x[0:3]
        v = x[3:6]
        beta = x[6]

        # compute acceleration derivatives
        dadr = self.dacc_dr(r, v, beta)
        dadv = self.dacc_dv(r, v, beta)
        dadbeta = self.dacc_dbeta(r, v, beta)
        # total derivative

        dfdx = np.zeros((7, 7))

        # d(r_k+1)/dr
        for i in range(3):
            for j in range(3):
                dfdx[i, j] = 0.5*pow(self.m.delta_t, 2)*dadr[i, j]

            dfdx
        for i in range(3):
            dfdx[i, i] = 1 + dfdx[i, i]

        # d(r_k+1)/dv
        for i in range(3):
            for j in range(3):
                dfdx[i, j+3] = 0.5*pow(self.m.delta_t, 2)*dadv[i, j]
        for i in range(3):
            dfdx[i, i+3] = self.m.delta_t + dfdx[i, i+3]

        # d(r_k+1)/dbeta
        for i in range(3):
            dfdx[i, 6] = 0.5*(self.m.delta_t**2)*dadbeta[i]

        # d(v_k+1)/dr
        for i in range(3):
            for j in range(3):
                dfdx[i+3, j] = self.m.delta_t*dadr[i, j]

        # d(v_k+1)/dv
        for i in range(3):
            for j in range(3):
                dfdx[i+3, j+3] = self.m.delta_t*dadv[i, j]
        for i in range(3):
            dfdx[i+3, i + 3] = 1 + dfdx[i+3, i + 3]

        # d(v_k+1)/dbeta
        for i in range(3):
            dfdx[i+3, 6] = self.m.delta_t*dadbeta[i]

        dfdx[6, 6] = 1
        return dfdx

    def dh(self, x):
        # x: point at which the derivative is evaluated
        r = self.o.position_transform(x[0:3])

        norm_r = LA.norm(r)
        dhdx = np.zeros((3, 7))
        dhdx[0, range(3)] = np.matmul(r, self.o.transform_M)/norm_r

        constant1 = 1/(np.sqrt(1-pow(r[2]/norm_r, 2)))
        constant2 = -constant1*r[2]*pow(norm_r, -3)
        dhdx[1, range(3)] = np.matmul([constant2*r[0], constant2*r[1], constant1*(pow(norm_r, -1) + constant2*r[2])], self.o.transform_M)

        constant3 = 1/((1 + pow(r[1]/r[0], 2))*r[0])
        dhdx[2, range(3)] = np.matmul([constant3*r[1]/r[0], -constant3, 0], self.o.transform_M)

        return dhdx

    def gradient(self, x):

        if len(x) != len(self.vars):
            var = np.ones((2*self.N+1)*7)
            selection = 'on'
            for i in range(0,2*self.N+1,2):
                var[i*7:(i+1)*7] = x[7*i//2:(1+i//2)*7]
        else:
            selection = 'off'
            var = np.copy(x)

        grad = np.zeros((1+2*self.N)*7)
        dh_i = []
        df_i = []
        f_i = []
        for i in range(0,2*self.N + 1,2):
            dh_i.append(self.dh(var[i*7:(i+1)*7]))
            df_i.append(self.dfdx(var[i*7:(i+1)*7]))
            r, v, a, beta = self.m.f(var[i*7:i*7+3], var[i*7+3:i*7+6], var[i*7+6], 'off')
            f_i.append([r[0], r[1], r[2], v[0], v[1], v[2], beta])

        R1 = np.zeros((3,3))
        R2 = np.zeros((7,7))
        for i in range(3):
            R1[i,:] = self.reg1[i,:]*self.reg1[i,i]
        for i in range(7):
            R2[i,:] = self.reg2[i,:]*self.reg2[i,i]
        R_mu = np.identity(7)
        for i in range(7): R_mu[i, i] = (self.reg2[i, i] / self.model_pen[i])**2

        # grad[0:7] = np.matmul(np.transpose(dh_i[0]), np.multiply(R1, self.o.h(var[0:3], 'off') - self.y[0])) \
        grad[0:7] = np.matmul(np.transpose(dh_i[0]), np.matmul(R1, self.o.h(var[0:3], 'off') - self.y[0])) \
                    - np.matmul(np.transpose(df_i[0]), np.matmul(R2, self.pen1*var[7:14]) + self.pen*np.matmul(R2, var[2*7:3*7] - f_i[0])) + \
                    np.matmul(R_mu, var[0:7] - self.x_prior)# dJ/dx

        for i in range(2,2*self.N,2):
            grad[(i-1)*7:i*7] = self.pen1*(np.matmul(R2, var[i*7:(i+1)*7] - f_i[i//2-1]))  # dJ/d(lambda)
            grad[i*7:(i+1)*7] = np.matmul(np.transpose(dh_i[i//2]), np.matmul(R1, self.o.h(var[i*7:i*7+3], 'off') - self.y[i//2])) \
                                - np.matmul(np.transpose(df_i[i//2]), self.pen1*var[(i+1)*7:(i+2)*7]) \
                                - np.matmul(np.transpose(df_i[i//2]), self.pen*np.matmul(R2, (var[(i+2)*7:(i+3)*7] - f_i[i//2]))) \
                                + self.pen*np.matmul(R2,(var[i*7:(i+1)*7] - f_i[i//2-1])) + self.pen1*var[(i-1)*7:i*7]  # dJ/d(lambda)

        grad[(2*self.N -1)*7:(2 * self.N )*7] = self.pen1*(var[2*self.N*7:2*self.N*7+7] - f_i[self.N-1])
        grad[(2*self.N)*7:(2*self.N+1)*7] = np.matmul(np.transpose(dh_i[self.N]),  np.matmul(R1, self.o.h(var[2*self.N*7:2*self.N*7+3], 'off') - self.y[self.N])) \
                                            + self.pen*np.matmul(R2, var[2*self.N*7:(2*self.N+1)*7] - f_i[self.N-1]) + self.pen1*var[(2*self.N-1)*7:(2*self.N)*7]

        # checking method
        # for l in range(len(var)):
        #     if (l//7)%2 == 0:
        #         eps = 0.1
        #         plus_eps = np.copy(var)
        #         plus_eps[l] = plus_eps[l]+eps
        #         minus_eps = np.copy(var)
        #         minus_eps[l] = minus_eps[l] - eps
        #
        #         A = self.cost(plus_eps)
        #         print('--')
        #         B = self.cost(minus_eps)
        #         # print(A)
        #         # print(B)
        #         derivative = (A-B)/(2*eps)
        #         print(l, ': diff (%): ', 100*np.abs(np.divide(grad[l]-derivative, grad[l], out=np.zeros_like(grad[l]), where=grad[l]!=0)))
        #         # print(l, ': diff (%): ', 100*np.abs(np.divide(grad[l]-derivative, derivative, out=np.zeros_like(derivative), where=derivative!=0)))
        #         print('  analytical: ', grad[l], '; numerical: ', derivative)
        # sys.exit()

        if selection == 'on':
            reduced_grad = np.ones((self.N+1)*7)
            for i in range(0,2*self.N+1,2):
                reduced_grad[7*i//2:(1+i//2)*7] = grad[i*7:(i+1)*7]
            return reduced_grad
        else:
            return grad




    def hessian(self, x):
        if len(x) != len(self.vars):
            var = np.ones((2*self.N+1)*7)
            selection = 'on'
            for i in range(0,2*self.N+1,2):
                var[i*7:(i+1)*7] = x[7*i//2:(1+i//2)*7]
        else:
            selection = 'off'
            var = np.copy(x)

        # The function assumes that the second derivatives of f and h are null
        H = np.zeros(((2*self.N+1)*7, (2*self.N+1)*7))
        R1 = np.zeros((3, 3))
        R2 = np.zeros((7, 7))
        for i in range(3):
            R1[i, :] = self.reg1[i, :] * self.reg1[i, i]
        for i in range(7):
            R2[i, :] = self.reg2[i, :] * self.reg2[i, i]
        R_mu = np.identity(7)
        for i in range(7): R_mu[i, i] = (self.reg2[i, i] / self.model_pen[i]) ** 2

        dhdx = self.dh(var[0:7])
        dfdx = self.dfdx(var[0:7])
        H[0:7, 0:7] = np.matmul(np.transpose(dhdx), np.matmul(R1, dhdx)) + self.pen*np.matmul(np.transpose(dfdx), np.matmul(R2, dfdx)) + self.mu*R_mu # dJ/d(x^2)
        H[0:7, 2*7:3*7] = -self.pen*np.matmul(np.transpose(dfdx), R2)  # dJ/d(x_i)d(x_i+1)
        H[0:7, 7:14] = -np.transpose(dfdx) * self.pen1  # dJ/dxd(lambda_i+1)
        H[7:14, 0:7] = -dfdx * self.pen1  # dJ/d(lambda)dx
        H[7:14, 7:14] = np.zeros((7, 7)) * self.pen1  # dJ/d(lambda^2)
        H[7:14, 14:21] = np.identity(7) * self.pen1  # dJ/d(lambda)d(x_i+1)

        for i in range(2,2*self.N,2): # it does not cover last point
            dhdx = self.dh(var[i*7:(i+1)*7])
            dfdx = self.dfdx(var[i*7:(i+1)*7])
            H[i*7:(i+1)*7, i*7:(i+1)*7] = np.matmul(np.transpose(dhdx), np.matmul(R1, dhdx)) + self.pen * np.matmul(np.transpose(dfdx), np.matmul(R2, dfdx)) \
                                          + self.pen*R2  # dJ/d(x_i^2)
            H[i*7:(i+1)*7, (i+2)*7:(i+3)*7] = -self.pen*np.matmul(np.transpose(dfdx), R2)  # dJ/d(x_i)d(x_i+1)
            H[i*7:(i+1)*7, (i-2)*7:(i-1)*7] = -self.pen*np.matmul(R2, self.dfdx(var[(i-2)*7:(i-1)*7])) # dJ/d(x_i)d(x_i-1)
            H[i*7:(i+1)*7, (i+1)*7:(i+2)*7] = -self.pen1*np.transpose(dfdx)  # dJ/dxd(lambda_i+1)
            H[i*7:(i+1)*7, (i-1)*7:i*7] = self.pen1*np.identity(7)  # dJ/dxd(lambda_i))
            H[(i+1)*7:(i+2)*7, (i+1)*7:(i+2)*7] = np.zeros((7, 7))  # dJ/d(lambda^2)
            H[(i+1)*7:(i+2)*7, i*7:(i+1)*7] = - dfdx * self.pen1 # dJ/d(lambda)dx
            H[(i+1)*7:(i+2)*7, (i+2)*7:(i+3)*7] = np.identity(7) * self.pen1  # dJ/d(lambda)d(x_i+1)

        dhdx = self.dh(var[(2*self.N)*7:(2*self.N+1)*7])
        H[(2*self.N)*7:(2*self.N+1)*7, (2*self.N)*7:(2*self.N+1)*7] = np.matmul(np.transpose(dhdx),np.matmul(R1, dhdx)) + self.pen * R2  # dJ/d(x^2)
        H[(2*self.N)*7:(2*self.N+1)*7, (self.N*2-2)*7:(self.N*2-1)*7] = -self.pen*np.matmul(R2, dfdx)  # dJ/d(x_i)d(x_i-1)
        H[(2*self.N)*7:(2*self.N+1)*7, (2*self.N-1)*7:(2*self.N)*7] = np.identity(7) * self.pen1  # dJ/dxd(lambda_i))

        # checking method
        # for l in range(len(var)):
        #     if (l // 7) % 2 == 0:
        #         eps = 100
        #
        #         plus_eps = np.copy(var)
        #         plus_eps[l] = plus_eps[l]+eps
        #         minus_eps = np.copy(var)
        #         minus_eps[l] = minus_eps[l] - eps
        #
        #         A = self.gradient(plus_eps)
        #         B = self.gradient(minus_eps)
        #         derivative = (A-B)/(2*eps)
        #         # H[l, :] = derivative
        #         print(l, ': max diff (%): ', 100*np.amax(np.abs(np.divide(H[l,: ] - derivative, H[l,: ], out=np.zeros_like(H[l,: ]), where=H[l,: ]!=0))))
        #         # print('numerical', derivative)
        #         # print('analytical', H[l, :])
        # sys.exit()

        if selection == 'on':
            reduced_H = np.ones(((self.N+1)*7,(self.N+1)*7))
            for i in range(0, 2 * self.N + 1, 2):
                for j in range(0, 2 * self.N + 1, 2):
                    reduced_H[(i // 2) * 7:(1 + i // 2) * 7, (j // 2) * 7:(1 + j // 2) * 7] = H[i * 7:(i + 1) * 7,
                                                                                            j * 7:(j + 1) * 7]
            return reduced_H
        else:
            return H

    def slide_window(self, last_y):
        # slide horizon of predicted states
        self.vars[0:(2*self.N-1)*7] = self.vars[2*7:(2*self.N+1)*7]
        self.vars[(2*self.N - 1)*7: 2*self.N*7] = np.ones(7)
        self.vars[2*self.N*7:2*self.N*7+3], self.vars[2*self.N*7+3:2*self.N*7+6], throw, self.vars[2*self.N*7+6] = \
            self.m.f(self.vars[2*self.N*7:2*self.N*7+3],self.vars[2*self.N*7+3:2*self.N*7+6],self.vars[2*self.N*7+6])

        # slide horizon of measurements
        self.y[0:self.N] = self.y[1:self.N+1]
        self.y[self.N] = last_y

        self.x_prior = self.vars[0:7]

        # it is assumed that the horizon is sufficiently small such that all measurements are of the same order at
        # end and beginning of the horizon
        # measurements reg

        for i in range(3):
            if np.abs(self.y[0, i]) < 1:
                mult = 1
                while mult * np.abs(self.y[0, i]) <= 1:
                    mult = mult * 10
                self.reg1[i,i] = self.measurement_pen[i] * mult
            else:
                mult = 1
                while np.abs(self.y[0, i]) // mult >= 10:
                    mult = mult * 10
                self.reg1[i,i] = self.measurement_pen[i] / mult

        # position and velocity reg
        for i in range(2):
            mult = 1
            if np.abs(self.vars[i * 3]) < 1:
                while mult * np.abs(self.vars[i * 3]) <= 1:
                    mult = mult * 10
                for j in range(3): self.reg2[i*3 + j, i*3 + j] =  self.model_pen[i*3+j] * mult
            else:
                while np.abs(self.vars[i * 3]) // mult >= 10:
                    mult = mult * 10
                for j in range(3): self.reg2[i*3 + j, i*3 + j] = self.model_pen[i*3+j] / mult

        # ballistic coeff reg
        mult = 1
        while np.abs(self.vars[6]) // mult >= 10:
            mult = mult * 10
        self.reg2[6, 6] = self.model_pen[6] / mult

        # self.reg1 = np.ones(3)
        # self.reg2 = np.ones(7)
        #
        # self.reg1 = np.multiply(self.reg1, self.measurement_pen)
        # self.reg2 = np.multiply(self.reg2, self.model_pen)

    def estimation(self, mea_pen=[], mod_pen=[]):
        # self.reg1 = mea_pen
        # self.reg2 = mod_pen

        grad = self.gradient(self.vars)
        hess = self.hessian(self.vars)

        if self.method == 'BFGS':
            self.vars = BFGS(self.vars, hess, self.cost, self.gradient, self.N)
        elif self.method == 'Newton LS':
            print(self.cost(self.vars))
            for i in range(20):
                self.vars = newton_iter_selection(self.vars, grad, hess, self.N, self.cost, 'on')
                grad = self.gradient(self.vars)
                hess = self.hessian(self.vars)
                # self.vars = newton_iter(self.vars, grad, hess)
            print(self.cost(self.vars))
        elif self.method == 'Newton':
            for i in range(150):
                self.vars = newton_iter_selection(self.vars, grad, hess, self.N, self.cost, 'off')
                grad = self.gradient(self.vars)
                hess = self.hessian(self.vars)
                # self.vars = newton_iter(self.vars, grad, hess)
        elif self.method == 'Gradient':
            self.vars = gradient_search(self.vars, self.cost, self.gradient)
        elif self.method == 'Built-in optimizer':
            # select vars
            vars_select = np.zeros((self.N+1)*7)
            for i in range(0, 2 * self.N + 1, 2):
                vars_select[(i // 2) * 7:(i // 2 + 1) * 7] = self.vars[i * 7:(i + 1) * 7]
            result = minimize(fun=self.cost, x0=vars_select, method='trust-ncg', jac=self.gradient, hess=self.hessian, options = {'maxiter': 50})

            for i in range(0, 2 * self.N + 1, 2):
                self.vars[i * 7:(i + 1) * 7] = result.x[(i // 2) * 7:(i // 2 + 1) * 7]
        else:
            print('Optimization method ' + self.method + ' non recognize')

    def tuning_MHE(self, real_x, real_beta, step):
        self.real_x = np.ones((1+2*self.N)*7)
        for i in range(0, 2*self.N+1, 2):
            for j in range(3):
                self.real_x[i*7+j] = real_x[step-self.N-1 + i//2][0][j]
            for j in range(3):
                self.real_x[i*7+j+3] = real_x[step-self.N-1 + i//2][1][j]
            self.real_x[i*7 + 6] = real_beta[i//2]
        coeffs = []
        for i in range(3): coeffs.append(self.measurement_pen[i])
        for i in range(7): coeffs.append(self.model_pen[i])

        tuning = minimize(self.tuning_cost, coeffs, method='Nelder-Mead', options = {'maxiter': 500} )
        tuning.x = np.abs(tuning.x)
        for i in range(3):
            self.reg1[i] = self.reg1[i]*tuning.x[i]/self.measurement_pen[i]
            self.measurement_pen[i] = tuning.x[i]
        for i in range(7):
            self.reg2[i] = self.reg2[i]*tuning.x[i+3]/self.model_pen[i]
            self.model_pen[i] = tuning.x[i+3]


    def tuning_cost(self, coeffs):
        reg1 = np.ones(3)
        reg2 = np.ones(7)
        for i in range(3):
            reg1[i] = self.reg1[i]*coeffs[i]/np.abs(self.measurement_pen[i])
        for i in range(7):
            reg2[i] = self.reg2[i]*coeffs[i+3]/np.abs(self.model_pen[i])

            a =1
        h_i = []
        f_i = np.zeros((self.N+1)*7)

        for i in range(0, 2*self.N+1, 2):
            h_i.append(self.o.h(self.real_x[i*7:i*7+3], 'off'))
            f_i[(i//2)*7:(i//2)*7+3], f_i[(i//2)*7+3:(i//2)*7+6], a, f_i[(i//2)*7+6] = self.m.f(self.real_x[i*7:i*7+3], self.real_x[i*7+3:i*7+6], self.real_x[i*7+6], 'off')

        J = 0

        for i in range(self.N + 1):
            J = J + 0.5 * LA.norm(np.multiply(reg1, self.y[i] - h_i[i]))**2

        for i in range(0, 2*self.N, 2):
            J = J + self.pen1*np.matmul(self.real_x[(i+1)*7:(i+2)*7], np.multiply(reg2, self.real_x[(i+2)*7:(i+3)*7] - f_i[(i//2)*7:(i//2+1)*7])) \
                + 0.5*self.pen*LA.norm(np.multiply(reg2, self.real_x[(i+2)*7:(i+3)*7] - f_i[(i//2)*7:(i//2+1)*7]))**2

        for i in range(10):
            J = J + np.abs(10/coeffs[i])
        return J


# # it is assumed that the horizon is sufficiently small such that all measurements are of the same order at
# # end and beginning of the horizon
# # measurements reg
#
# for i in range(3):
#     if np.abs(self.y[0, i]) < 1:
#         mult = 1
#         while mult * np.abs(self.y[0, i]) <= 1:
#             mult = mult * 10
#         self.reg1[i] = 1 * mult
#     else:
#         mult = 1
#         while np.abs(self.y[0, i]) // mult >= 10:
#             mult = mult * 10
#         self.reg1[i] = 1 / mult
#
# # position and velocity reg
# for i in range(2):
#     mult = 1
#     if np.abs(self.vars[i * 3]) < 1:
#         while mult * np.abs(self.vars[i * 3]) <= 1:
#             mult = mult * 10
#         self.reg2[i * 3:i * 3 + 3] = 1 * mult
#     else:
#         while np.abs(self.vars[i * 3]) // mult >= 10:
#             mult = mult * 10
#         self.reg2[i * 3:i * 3 + 3] = 1 / mult
#
# # ballistic coeff reg
# mult = 1
# while np.abs(self.vars[6]) // mult >= 10:
#     mult = mult * 10
# self.reg2[6] = 1 / mult




