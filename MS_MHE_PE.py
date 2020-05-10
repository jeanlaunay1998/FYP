from numpy import linalg as LA
import numpy as np
from newton_method import newton_iter_selection
from newton_method import BFGS
from filterpy.kalman import ExtendedKalmanFilter as EKF
from newton_method import newton_iter
import sys
from scipy.optimize import minimize

class MS_MHE_PE:
    def __init__(self, estimation_model, true_system, observer, horizon, measurement_lapse, pen1, pen2, ekf_parameters, opt_method='Newton'):
        self.N = horizon
        self.m = estimation_model
        self.d = true_system
        self.o = observer
        self.method = opt_method
        self.inter_steps = int(measurement_lapse / self.m.delta_t)

        self.P0 = ekf_parameters[0]
        self.P_end = []
        self.Q = ekf_parameters[1]
        self.R = ekf_parameters[2]
        self.ekf = EKF(dim_x=7, dim_z=3)

        # the state x = {r, v, beta}
        self.vars = np.ones((1+self.N)*7) # list of inputs to the optimization
        self.K_horizon = []
        self.P_horizon = []
        self.y = []  # list of true measurements across the horizon
        self.pen = 1  # penalty factor
        self.pen1 = 0  # factor to remove Lagrangians (note this is highly inefficient since there will be un-used variables)
        self.pen2 = 0  # 10e-5

        self.reg1 = np.zeros(3)  # distance, azimuth, elevation
        self.reg2 = np.zeros(7)  # position, velocity and ballistic coeff

        self.measurement_pen = pen1
        self.model_pen = pen2


    def estimator_initilisation(self, step, y_measured):
        # this function initiates the estimator
        self.y = np.array(y_measured)[step - self.N - 1:step, :]

        # first point is given by the model
        x0 = np.zeros(7)
        x0[0:3] = np.copy(self.m.Sk[len(self.m.Sk) - (1 + self.N)*self.inter_steps][0][0:3])
        x0[3:6] = np.copy(self.m.Sk[len(self.m.Sk) - (1 + self.N)*self.inter_steps][1][0:3])
        x0[6] = self.m.beta

        self.ekf.x = x0
        self.vars[0:7] = x0
        self.ekf.P = np.copy(self.P0)
        self.ekf.Q = self.Q
        self.ekf.R = self.R
        for i in range(1, self.N+1):
            self.ekf.F = self.dfdx(self.ekf.x)
            self.ekf.predict(fx=self.m.f)
            self.ekf.update(z=self.y[i], HJacobian=self.dh, Hx=self.o.h)
            self.vars[i*7:(i+1)*7] = self.ekf.x
            self.P_horizon.append(self.ekf.P)
            self.K_horizon.append(self.ekf.K)
        self.P_end = self.ekf.P

        self.reg1 = np.ones(3)
        self.reg2 = np.ones(7)

        self.reg1 = np.multiply(self.reg1, self.measurement_pen)
        self.reg2 = np.multiply(self.reg2, self.model_pen)


    def cost(self, var):

        h_i = []
        g_i = np.zeros((self.N+1)*7)

        self.ekf.P = np.copy(self.P0)
        self.ekf.x = var[0:7]
        # for i in range(1, self.N+1):
        #     self.ekf.F = self.dfdx(self.ekf.x)
        #     self.ekf.predict(fx=self.m.f)
        #     self.ekf.update(z=self.y[i], HJacobian=self.dh, Hx=self.o.h)
        #     a = self.ekf.x
        #     print(a)

        for i in range(0,self.N):
            # self.ekf.x = var[i*7:(i+1)*7]
            self.ekf.F = self.dfdx(self.ekf.x)
            self.ekf.predict(fx=self.m.f)
            self.ekf.update(z=self.y[i+1], HJacobian=self.dh, Hx=self.o.h)
            g_i[i*7:(i+1)*7] = self.ekf.x
            h_i.append(self.o.h(var[i*7:i*7+3], 'off'))
        h_i.append(self.o.h(var[self.N*7:self.N*7+3], 'off'))

        J = 0
        for i in range(self.N + 1):
            J = J + 0.5 * LA.norm(np.multiply(self.reg1, self.y[i] - h_i[i]))**2

        for i in range(self.N):
            J = J + 0.5*self.pen*LA.norm(np.multiply(self.reg2, var[(i+1)*7:(i+2)*7] - g_i[i*7:(i+1)*7]))**2
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

    def dgdx(self, x, P_i, y_i):

        self.ekf.P = P_i
        self.ekf.x = x
        df = self.dfdx(self.ekf.x)
        self.ekf.F = df
        self.ekf.predict(fx=self.m.f)
        self.ekf.update(z=y_i, HJacobian=self.dh, Hx=self.o.h)
        x_aprior = np.copy(self.ekf.x)
        P_return = np.copy(self.ekf.P)
        dg = np.matmul(np.identity(7) - np.matmul(self.ekf.K, self.dh(x_aprior)), df)

        q = y_i - self.o.h(x_aprior, 'off')
        # checking method
        for i in range(7):
            eps = 0.001
            # ------------#
            plus_eps = np.copy(x)
            plus_eps[i] = plus_eps[i]+eps
            self.ekf.P = P_i
            self.ekf.x = plus_eps
            self.ekf.F = self.dfdx(self.ekf.x)
            self.ekf.predict(fx=self.m.f)
            self.ekf.update(z=y_i, HJacobian=self.dh, Hx=self.o.h)
            # A = self.ekf.x
            K_plus = self.ekf.K
            # ------------#
            plus_eps = np.copy(x)
            plus_eps[i] = plus_eps[i] - eps
            self.ekf.P = P_i
            self.ekf.x = plus_eps
            self.ekf.F = self.dfdx(self.ekf.x)
            self.ekf.predict(fx=self.m.f)
            self.ekf.update(z=y_i, HJacobian=self.dh, Hx=self.o.h)
            # B = self.ekf.x
            dK = (K_plus - self.ekf.K)/(2*eps)
            # derivative = (A-B)/(2*eps)
            dg[:,i] = dg[:,i] + np.matmul(dK, q)

            # print(i, ': max difference: ', 100*np.amax(np.abs(np.divide(derivative - dg[:, i], dg[:, i], out=np.zeros_like(dg[:, i]), where=dg[:, i] != 0))))
            # print('-----')
        return [dg, x_aprior, P_return]



    def gradient(self, var):
        grad = np.zeros((1+self.N)*7)
        dh_i = []
        dg_i = []
        g_i = []
        self.ekf.P = np.copy(self.P0)
        for i in range(self.N):
            results = self.dgdx(x=var[i*7:(i+1)*7], P_i=self.ekf.P, y_i=self.y[i+1])
            dg_i.append(results[0])
            g_i.append(results[1])
            self.ekf.P = results[2]
            dh_i.append(self.dh(var[i*7:(i+1)*7]))
        dh_i.append(self.dh(var[self.N*7:(self.N+1)*7]))

        R1 = np.power(self.reg1, 2)
        R2 = np.power(self.reg2, 2)

        grad[0:7] = np.matmul(np.transpose(dh_i[0]), np.multiply(R1, self.o.h(var[0:3], 'off') - self.y[0])) \
                    - np.matmul(np.transpose(dg_i[0]), self.pen*np.multiply(R2, var[7:2*7] - g_i[0]))  # dJ/dx

        for i in range(1,self.N):
            grad[i*7:(i+1)*7] = np.matmul(np.transpose(dh_i[i]), np.multiply(R1, self.o.h(var[i*7:i*7+3], 'off') - self.y[i])) \
                                - np.matmul(np.transpose(dg_i[i]), self.pen*np.multiply(R2, var[(i+1)*7:(i+2)*7] - g_i[i]))\
                                + self.pen*np.multiply(R2,var[i*7:(i+1)*7] - g_i[i-1])
        grad[self.N*7:(self.N+1)*7] = np.matmul(np.transpose(dh_i[self.N]),  np.multiply(R1, self.o.h(var[self.N*7:self.N*7+3], 'off') - self.y[self.N])) \
                                            + self.pen*np.multiply(R2, var[self.N*7:(self.N+1)*7] - g_i[self.N-1])

        # checking method
        # for l in range(len(var)):
        #     eps = 0.1
        #     plus_eps = np.copy(var)
        #     plus_eps[l] = plus_eps[l]+eps
        #     minus_eps = np.copy(var)
        #     minus_eps[l] = minus_eps[l] - eps
        #
        #     A = self.cost(plus_eps)
        #     print('--')
        #     B = self.cost(minus_eps)
        #     derivative = (A-B)/(2*eps)
        #     print(l, ': diff (%): ', 100*np.abs(np.divide(grad[l]-derivative, grad[l], out=np.zeros_like(grad[l]), where=grad[l]!=0)))
        #     print('  analytical: ', grad[l], '; numerical: ', derivative)

        return grad




    def hessian(self, var):

        # The function assumes that the second derivatives of f and h are null
        H = np.zeros(((self.N+1)*7, (self.N+1)*7))
        R1 = np.multiply(np.identity(3), np.power(self.reg1, 2))
        R2 = np.multiply(np.identity(7), np.power(self.reg2, 2))

        self.ekf.P = np.copy(self.P0)
        results = self.dgdx(x=var[0:7], P_i=self.ekf.P, y_i=self.y[1])
        self.ekf.P = results[2]
        dgdx = results[0]
        dhdx = self.dh(var[0:7])
        dgdx_minus = dgdx

        H[0:7, 0:7] = np.matmul(np.transpose(dhdx), np.matmul(R1, dhdx)) + self.pen*np.matmul(np.transpose(dgdx), np.matmul(R2, dgdx))  # dJ/d(x^2)
        H[0:7, 2*7:3*7] = -self.pen*np.matmul(np.transpose(dgdx), R2)  # dJ/d(x_i)d(x_i+1)

        for i in range(1, self.N):  # it does not cover last point
            results = self.dgdx(x=var[i*7:(i+1)*7], P_i=self.ekf.P, y_i=self.y[i+1])
            self.ekf.P = results[2]
            dgdx = results[0]
            dhdx = self.dh(var[i*7:(i+1)*7])
            H[i*7:(i+1)*7, i*7:(i+1)*7] = np.matmul(np.transpose(dhdx), np.matmul(R1, dhdx)) + self.pen * np.matmul(np.transpose(dgdx), np.matmul(R2, dgdx)) \
                                          + self.pen*R2  # dJ/d(x_i^2)
            H[i*7:(i+1)*7, (i+1)*7:(i+2)*7] = -self.pen*np.matmul(np.transpose(dgdx), R2)  # dJ/d(x_i)d(x_i+1)
            H[i*7:(i+1)*7, (i-1)*7:i*7] = -self.pen*np.matmul(R2, dgdx_minus) # dJ/d(x_i)d(x_i-1)
            dgdx_minus = dgdx

        dhdx = self.dh(var[self.N*7:(self.N+1)*7])
        H[self.N*7:(self.N+1)*7, self.N*7:(self.N+1)*7] = np.matmul(np.transpose(dhdx),np.matmul(R1, dhdx)) + self.pen * R2  # dJ/d(x^2)
        H[self.N*7:(self.N+1)*7, (self.N-1)*7:self.N*7] = -self.pen*np.matmul(R2, dgdx)  # dJ/d(x_i)d(x_i-1)

        # checking method
        # for l in range(len(var)):
        #     eps = 100
        #
        #     plus_eps = np.copy(var)
        #     plus_eps[l] = plus_eps[l]+eps
        #     minus_eps = np.copy(var)
        #     minus_eps[l] = minus_eps[l] - eps
        #
        #     A = self.gradient(plus_eps)
        #     B = self.gradient(minus_eps)
        #     derivative = (A-B)/(2*eps)
        #     # H[l, :] = derivative
        #     print(l, ': max diff (%): ', 100*np.amax(np.abs(np.divide(H[l,: ] - derivative, H[l,: ], out=np.zeros_like(H[l,: ]), where=H[l,: ]!=0))))
        #     # print('numerical', derivative)
        #     # print('analytical', H[l, :])
        return H


    def slide_window(self, last_y):
        # slide horizon of measurements
        self.y[0:self.N] = self.y[1:self.N+1]
        self.y[self.N] = last_y

        # slide horizon of predicted states
        self.vars[0:self.N * 7] = self.vars[7:(self.N + 1) * 7]

        self.ekf.P = self.P_end
        self.ekf.x = self.vars[self.N*7:(self.N+1)*7]
        # print(self.ekf.x)

        self.ekf.F = self.dfdx(self.ekf.x)
        self.ekf.predict(fx=self.m.f)
        self.ekf.update(z=last_y, HJacobian=self.dh, Hx=self.o.h)
        self.vars[self.N * 7:self.N * 7 + 7] = self.ekf.x # new value obtained at t = k+1  from x_k+1 = g(x_sol_k)

        self.reg1 = np.ones(3)
        self.reg2 = np.ones(7)

        self.reg1 = np.multiply(self.reg1, self.measurement_pen)
        self.reg2 = np.multiply(self.reg2, self.model_pen)
        ################################################ we need to change self.P_end at some point for the next iteration

    def estimation(self):

        grad = self.gradient(self.vars)
        hess = self.hessian(self.vars)

        if self.method == 'BFGS':
            self.vars = BFGS(self.vars, hess, self.cost, self.gradient, self.N)
        elif self.method == 'Newton LS':
            for i in range(2):
                print(self.cost(self.vars))
                self.vars = newton_iter_selection(self.vars, grad, hess, self.N, self.cost, 'on')
                print(self.cost(self.vars))
                grad = self.gradient(self.vars)
                hess = self.hessian(self.vars)
                # self.vars = newton_iter(self.vars, grad, hess)
        elif self.method == 'Newton':
            for i in range(10):
                self.vars = newton_iter_selection(self.vars, grad, hess, self.N, self.cost, 'off')
                grad = self.gradient(self.vars)
                hess = self.hessian(self.vars)
                # self.vars = newton_iter(self.vars, grad, hess)
        elif self.method == 'Built-in optimizer':
            # select vars
            # vars_select = np.zeros((self.N+1)*7)
            # for i in range(0, 2 * self.N + 1, 2):
            #     vars_select[(i // 2) * 7:(i // 2 + 1) * 7] = self.vars[i * 7:(i + 1) * 7]
            vars_select = self.vars
            result = minimize(fun=self.cost, x0=vars_select, method='trust-ncg', jac=self.gradient, hess=self.hessian, options = {'maxiter': 10})
            self.vars = result
            # for i in range(0, 2 * self.N + 1, 2):
            #     self.vars[i * 7:(i + 1) * 7] = result.x[(i // 2) * 7:(i // 2 + 1) * 7]
        else:
            print('Optimization method ' + self.method + ' non recognize')








