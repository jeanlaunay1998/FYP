from numpy import linalg as LA
import numpy as np
from newton_method import newton_iter_selection
from newton_method import newton_iter
import sys
from scipy.optimize import minimize

class multishooting:
    def __init__(self, estimation_model, true_system, observer, horizon, measurement_lapse):
        self.N = horizon
        self.m = estimation_model
        self.d = true_system
        self.o = observer

        self.inter_steps = int(measurement_lapse / self.m.delta_t)

        # the state x = {r, v, beta}
        self.vars = np.ones((1+2*self.N)*7) # list of inputs to the optimization
        self.y = []  # list of true measurements across the horizon
        self.pen = 1  # penalty factor
        self.pen1 = 0  # factor to remove Lagrangians (note this is highly inefficient since there will be un-used variables)
        self.pen2 = 0  # 10e-5

        self.reg1 = np.zeros(3)  # distance, azimuth, elevation
        self.reg2 = np.zeros(7)  # position, velocity and ballistic coeff

        self.measurement_pen = [1e6, 1e-1, 1e-1]
        self.model_pen = [1e4, 1e4, 1e4, 1e1, 1e1, 1e1, 1e0]


    def estimator_initilisation(self, step, y_measured):
        # this function initiates the estimator
        self.y = np.array(y_measured)[step - self.N - 1:step, :]
        for i in range(0, 2*self.N+1, 2):
            for j in range(3):
                self.vars[i*7+j] = np.copy(self.m.Sk[len(self.m.Sk)-(1+self.N)*self.inter_steps + i//2][0][j])
            for j in range(3,6):
                self.vars[i * 7 + j] = np.copy(self.m.Sk[len(self.m.Sk) - (1 + self.N) * self.inter_steps + i//2][1][j-3])
            self.vars[i*7 + 6] = self.m.beta

            # it is assumed that the horizon is sufficiently small such that all measurements are of the same order at
            # end and beginning of the horizon
            # measurements reg

            for i in range(3):
                if np.abs(self.y[0, i]) < 1:
                    mult = 1
                    while mult * np.abs(self.y[0, i]) <= 1:
                        mult = mult * 10
                    self.reg1[i] = 1 * mult
                else:
                    mult = 1
                    while np.abs(self.y[0, i]) // mult >= 10:
                        mult = mult * 10
                    self.reg1[i] = 1 / mult

            # position and velocity reg
            for i in range(2):
                mult = 1
                if np.abs(self.vars[i * 3]) < 1:
                    while mult * np.abs(self.vars[i * 3]) <= 1:
                        mult = mult * 10
                    self.reg2[i * 3:i * 3 + 3] = 1 * mult
                else:
                    while np.abs(self.vars[i * 3]) // mult >= 10:
                        mult = mult * 10
                    self.reg2[i * 3:i * 3 + 3] = 1 / mult

            # ballistic coeff reg
            mult = 1
            while np.abs(self.vars[6]) // mult >= 10:
                mult = mult * 10
            self.reg2[6] = 1 / mult

            # self.reg1 = np.ones(3)
            # self.reg2 = np.ones(7)

            self.reg1 = np.multiply(self.reg1, self.measurement_pen)
            self.reg2 = np.multiply(self.reg2, self.model_pen)


    def cost(self, var):
        J = 0
        h_i = []
        f_i = np.zeros((self.N+1)*7)

        for i in range(0, 2*self.N+1, 2):
            h_i.append(self.o.h(var[i*7:i*7+3], 'off'))
            f_i[(i//2)*7:(i//2)*7+3], f_i[(i//2)*7+3:(i//2)*7+6], a, f_i[(i//2)*7+6] = self.m.f(var[i*7:i*7+3], var[i*7+3:i*7+6], var[i*7+6], 'off')

        # J = 0.5*LA.norm(self.y-h_i)**2
        J = 0

        for i in range(self.N + 1):
            J = J + 0.5 * LA.norm(np.multiply(self.reg1, self.y[i] - h_i[i]))**2

        for i in range(0, 2*self.N, 2):
            J = J + self.pen1*np.matmul(var[(i+1)*7:(i+2)*7], np.multiply(self.reg2, var[(i+2)*7:(i+3)*7] - f_i[(i//2)*7:(i//2+1)*7])) \
                + 0.5*self.pen*LA.norm(np.multiply(self.reg2, var[(i+2)*7:(i+3)*7] - f_i[(i//2)*7:(i//2+1)*7]))**2 \
                + 0.5 * self.pen2*(var[i*7+6] - self.m.beta)**2
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

    def gradient(self, var):
        grad = np.zeros((1+2*self.N)*7)
        dh_i = []
        df_i = []
        f_i = []
        for i in range(0,2*self.N + 1,2):
            dh_i.append(self.dh(var[i*7:(i+1)*7]))
            df_i.append(self.dfdx(var[i*7:(i+1)*7]))
            r, v, a, beta = self.m.f(var[i*7:i*7+3], var[i*7+3:i*7+6], var[i*7+6], 'off')
            f_i.append([r[0], r[1], r[2], v[0], v[1], v[2], beta])

        R1 = np.power(self.reg1, 2)
        R2 = np.power(self.reg2, 2)

        # grad[0:7] = np.matmul(np.transpose(dh_i[0]), np.multiply(R1, self.o.h(var[0:3], 'off') - self.y[0])) \
        grad[0:7] = np.matmul(np.transpose(dh_i[0]), np.multiply(R1, self.o.h(var[0:3], 'off') - self.y[0])) \
                    - np.matmul(np.transpose(df_i[0]), np.multiply(R2, self.pen1*var[7:14]) + self.pen*np.multiply(R2, var[2*7:3*7] - f_i[0]))  # dJ/dx
        grad[6] = grad[6] + self.pen2*(var[6] - self.m.beta)

        for i in range(2,2*self.N,2):
            grad[(i-1)*7:i*7] = self.pen1*(np.multiply(R2, var[i*7:(i+1)*7] - f_i[i//2-1]))  # dJ/d(lambda)
            grad[i*7:(i+1)*7] = np.matmul(np.transpose(dh_i[i//2]), np.multiply(R1, self.o.h(var[i*7:i*7+3], 'off') - self.y[i//2])) \
                                - np.matmul(np.transpose(df_i[i//2]), self.pen1*var[(i+1)*7:(i+2)*7]) \
                                - np.matmul(np.transpose(df_i[i//2]), self.pen*np.multiply(R2, (var[(i+2)*7:(i+3)*7] - f_i[i//2])))\
                                + self.pen*np.multiply(R2,(var[i*7:(i+1)*7] - f_i[i//2-1])) + self.pen1*var[(i-1)*7:i*7]  # dJ/d(lambda)
            grad[i*7+6] = grad[i*7+6] + self.pen2*(var[i*7+6]- self.m.beta)

        grad[(2*self.N -1)*7:(2 * self.N )*7] = self.pen1*(var[2*self.N*7:2*self.N*7+7] - f_i[self.N-1])
        grad[(2*self.N)*7:(2*self.N+1)*7] = np.matmul(np.transpose(dh_i[self.N]),  np.multiply(R1, self.o.h(var[2*self.N*7:2*self.N*7+3], 'off') - self.y[self.N])) \
                                            + self.pen*np.multiply(R2, var[2*self.N*7:(2*self.N+1)*7] - f_i[self.N-1]) + self.pen1*var[(2*self.N-1)*7:(2*self.N)*7]
        grad[2*self.N*7+6] = grad[2*self.N*7+6] + self.pen2*(var[2*self.N*7+6] - self.m.beta)

        # checking method
        # for l in range(len(var)):
        #     if (l//7)%2 == 0:
        #         eps = 1
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
        #         # print('numerical', derivative, 'analytical', grad[l])
        #         print(l, ': diff (%): ', 100*np.abs(np.divide(grad[l]-derivative, grad[l], out=np.zeros_like(grad[l]), where=grad[l]!=0)))
        #         # print(l, ': diff (%): ', 100*np.abs(np.divide(grad[l]-derivative, derivative, out=np.zeros_like(derivative), where=derivative!=0)))
        #         print('  analytical: ', grad[l], '; numerical: ', derivative)
        #
        # sys.exit()

        return grad


    def hessian(self, var):
        # The function assumes that the second derivatives of f and h are null
        H = np.zeros(((2*self.N+1)*7, (2*self.N+1)*7))
        R1 = np.multiply(np.identity(3), np.power(self.reg1, 2))
        R2 = np.multiply(np.identity(7), np.power(self.reg2, 2))

        dhdx = self.dh(var[0:7])
        dfdx = self.dfdx(var[0:7])
        H[0:7, 0:7] = np.matmul(np.transpose(dhdx), np.matmul(R1, dhdx)) + self.pen*np.matmul(np.transpose(dfdx), np.matmul(R2, dfdx))  # dJ/d(x^2)
        H[6, 6] = H[6, 6] + self.pen2 # add beta penalty
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
            H[i*7+6, i*7+6] = H[i*7+6, i*7+6] + self.pen2  # add penalty derivative
            H[i*7:(i+1)*7, (i+2)*7:(i+3)*7] = -self.pen*np.matmul(np.transpose(dfdx), R2)  # dJ/d(x_i)d(x_i+1)
            H[i*7:(i+1)*7, (i-2)*7:(i-1)*7] = -self.pen*np.matmul(R2, self.dfdx(var[(i-2)*7:(i-1)*7])) # dJ/d(x_i)d(x_i-1)
            H[i*7:(i+1)*7, (i+1)*7:(i+2)*7] = -self.pen1*np.transpose(dfdx)  # dJ/dxd(lambda_i+1)
            H[i*7:(i+1)*7, (i-1)*7:i*7] = self.pen1*np.identity(7)  # dJ/dxd(lambda_i))
            H[(i+1)*7:(i+2)*7, (i+1)*7:(i+2)*7] = np.zeros((7, 7))  # dJ/d(lambda^2)
            H[(i+1)*7:(i+2)*7, i*7:(i+1)*7] = - dfdx * self.pen1 # dJ/d(lambda)dx
            H[(i+1)*7:(i+2)*7, (i+2)*7:(i+3)*7] = np.identity(7) * self.pen1  # dJ/d(lambda)d(x_i+1)

        dhdx = self.dh(var[(2*self.N)*7:(2*self.N+1)*7])
        H[(2*self.N)*7:(2*self.N+1)*7, (2*self.N)*7:(2*self.N+1)*7] = np.matmul(np.transpose(dhdx),np.matmul(R1, dhdx)) + self.pen * R2  # dJ/d(x^2)
        H[(2*self.N)*7+6, (2*self.N)*7+6] = H[(2*self.N)*7+6, (2*self.N)*7+6] + self.pen2  # add beta penalty
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

        # it is assumed that the horizon is sufficiently small such that all measurements are of the same order at
        # end and beginning of the horizon
        # measurements reg

        for i in range(3):
            if np.abs(self.y[0, i]) < 1:
                mult = 1
                while mult * np.abs(self.y[0, i]) <= 1:
                    mult = mult * 10
                self.reg1[i] = 1 * mult
            else:
                mult = 1
                while np.abs(self.y[0, i]) // mult >= 10:
                    mult = mult * 10
                self.reg1[i] = 1 / mult

        # position and velocity reg
        for i in range(2):
            mult = 1
            if np.abs(self.vars[i * 3]) < 1:
                while mult * np.abs(self.vars[i * 3]) <= 1:
                    mult = mult * 10
                self.reg2[i * 3:i * 3 + 3] = 1 * mult
            else:
                while np.abs(self.vars[i * 3]) // mult >= 10:
                    mult = mult * 10
                self.reg2[i * 3:i * 3 + 3] = 1 / mult

        # ballistic coeff reg
        mult = 1
        while np.abs(self.vars[6]) // mult >= 10:
            mult = mult * 10
        self.reg2[6] = 1 / mult

        # self.reg1 = np.ones(3)
        # self.reg2 = np.ones(7)

        self.reg1 = np.multiply(self.reg1, self.measurement_pen)
        self.reg2 = np.multiply(self.reg2, self.model_pen)




    def estimation(self):
        cost_after = self.cost(self.vars)
        cost_before = cost_after + 10
        Niter = 0
        # while np.abs(cost_before - cost_after)> 10e-2:
        for i in range(2):
            Niter = Niter + 1
            cost_before = cost_after
            # print('before')
            # print(cost_before)
            grad = self.gradient(self.vars)
            hess = self.hessian(self.vars)
            # self.vars = newton_iter(self.vars, grad, hess)
            self.vars = newton_iter_selection(self.vars, grad, hess, cost_before, self.N)
            cost_after = self.cost(self.vars)
            # print('after')
            # print(cost_after)

            # if cost_after > cost_before:
            #     hessian = np.zeros(((self.N + 1) * 7, (self.N + 1) * 7))
            #     for l in range(0, 2 * self.N + 1, 2):
            #         for j in range(0, 2 * self.N + 1, 2):
            #             hessian[(l // 2) * 7:(1 + l // 2) * 7, (j // 2) * 7:(1 + j // 2) * 7] = hess[l * 7:(l + 1) * 7, j * 7:(j + 1) * 7]
            #     print('error starts here!!')
            # print(LA.norm(hessian))
            # for j in range(2, 2*self.N,2): print(grad[j*7:(j+1)*7])
            # for j in range((self.N+1)*7): print(hessian[j,j])


        print(Niter)






