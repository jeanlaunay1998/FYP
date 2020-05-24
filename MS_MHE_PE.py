from numpy import linalg as LA
import numpy as np
from newton_method import newton_iter_selection
from newton_method import BFGS
from newton_method import gradient_search
from extendedKF import ExtendedKalmanFilter as EKF
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

        self.reg1 = np.identity(3)  # distance, azimuth, elevation
        self.reg1 = LA.inv(np.array([[50, 0, 0], [0, (1e-2), 0], [0, 0, (1e-2)]])) # np.power(LA.inv(self.R),0.5) # np.zeros(3)  # distance, azimuth, elevation
        self.reg2 = np.identity(7)  # position, velocity and ballistic coeff
        self.measurement_pen = pen1
        self.model_pen = pen2
        for i in range(7):
            self.reg2[i,i] = self.model_pen[i]
        # VARIABLES FOR ARRIVAL COST
        self.x_prior = np.zeros(7)
        self.mu = 0 # 1e1


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
        self.x_prior = self.vars[0:7]


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

        R_mu = np.identity(7)
        for i in range(7): R_mu[i, i] = self.reg2[i, i] / self.model_pen[i]

        # Arrival cost
        J = 0.5 * self.mu * LA.norm(np.matmul(R_mu, self.vars[0:7] - self.x_prior)) ** 2

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
        for i in range(6): dg[6, i] = 0

        # checking method
        q = y_i - self.o.h(x_aprior, 'off')
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
            K_plus = self.ekf.K
            # A = self.ekf.x
            # ------------#
            plus_eps = np.copy(x)
            plus_eps[i] = plus_eps[i] - eps
            self.ekf.P = P_i
            self.ekf.x = plus_eps
            self.ekf.F = self.dfdx(self.ekf.x)
            self.ekf.predict(fx=self.m.f)
            self.ekf.update(z=y_i, HJacobian=self.dh, Hx=self.o.h)
            # B = self.ekf.x
            # derivative = (A-B)/(2*eps)
            dK = (K_plus - self.ekf.K)/(2*eps)
            new = dg[:,i] + np.matmul(dK, q)
            for l in range(6): dg[l,i] = new[l]

            # print(i, ': max difference: ', 100*np.amax(np.abs(np.divide(derivative - dg[:, i], dg[:, i], out=np.zeros_like(dg[:, i]), where=dg[:, i] != 0))))
            # print(dg[:,i])
            # print(derivative)
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

        R1 = np.zeros((3,3))
        R2 = np.zeros((7,7))
        for i in range(3):
            R1[i,:] = self.reg1[i,:]*self.reg1[i,i]
        for i in range(7):
            R2[i,:] = self.reg2[i,:]*self.reg2[i,i]
        R_mu = np.identity(7)
        for i in range(7): R_mu[i, i] = self.reg2[i, i] #/ self.model_pen[i]) ** 2

        grad[0:7] = np.matmul(np.transpose(dh_i[0]), np.matmul(R1, self.o.h(var[0:3], 'off') - self.y[0])) \
                    - np.matmul(np.transpose(dg_i[0]), self.pen*np.matmul(R2, var[7:2*7] - g_i[0]))  + \
                    np.matmul(R_mu, var[0:7] - self.x_prior)# dJ/dx

        for i in range(1,self.N):
            grad[i*7:(i+1)*7] = np.matmul(np.transpose(dh_i[i]), np.matmul(R1, self.o.h(var[i*7:i*7+3], 'off') - self.y[i])) \
                                - np.matmul(np.transpose(dg_i[i]), self.pen*np.matmul(R2, var[(i+1)*7:(i+2)*7] - g_i[i]))\
                                + self.pen*np.matmul(R2,var[i*7:(i+1)*7] - g_i[i-1])
        grad[self.N*7:(self.N+1)*7] = np.matmul(np.transpose(dh_i[self.N]),  np.matmul(R1, self.o.h(var[self.N*7:self.N*7+3], 'off') - self.y[self.N])) \
                                            + self.pen*np.matmul(R2, var[self.N*7:(self.N+1)*7] - g_i[self.N-1])

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
        #
        # sys.exit()

        return grad


    def hessian(self, var):

        # The function assumes that the second derivatives of f and h are null
        H = np.zeros(((self.N+1)*7, (self.N+1)*7))
        R1 = np.zeros((3,3))
        R2 = np.zeros((7,7))
        for i in range(3):
            R1[i,:] = self.reg1[i,:]*self.reg1[i,i]
        for i in range(7):
            R2[i,:] = self.reg2[i,:]*self.reg2[i,i]
        R_mu = np.identity(7)
        for i in range(7): R_mu[i, i] = self.reg2[i, i] #/ self.model_pen[i]) ** 2

        self.ekf.P = np.copy(self.P0)
        results = self.dgdx(x=var[0:7], P_i=self.ekf.P, y_i=self.y[1])
        self.ekf.P = results[2]
        dgdx = results[0]
        dhdx = self.dh(var[0:7])
        dgdx_minus = dgdx

        H[0:7, 0:7] = np.matmul(np.transpose(dhdx), np.matmul(R1, dhdx)) + self.pen*np.matmul(np.transpose(dgdx), np.matmul(R2, dgdx)) + self.mu*R_mu # dJ/d(x^2)
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
        self.P_end, self.P0 = self.last_covariance()
        # slide horizon of measurements
        self.y[0:self.N] = self.y[1:self.N+1]
        self.y[self.N] = last_y

        # slide horizon of predicted states
        self.vars[0:self.N * 7] = self.vars[7:(self.N + 1) * 7]

        self.ekf.P = self.P_end
        self.ekf.x = self.vars[self.N*7:(self.N+1)*7]
        self.ekf.F = self.dfdx(self.ekf.x)
        self.ekf.predict(fx=self.m.f)
        self.ekf.update(z=last_y, HJacobian=self.dh, Hx=self.o.h)
        self.vars[self.N * 7:self.N * 7 + 7] = self.ekf.x # new value obtained at t = k+1  from x_k+1 = g(x_sol_k)
        self.x_prior = self.vars[0:7]

    def estimation(self):
        grad = self.gradient(self.vars)
        hess = self.hessian(self.vars)

        if self.method == 'BFGS':
            self.vars = BFGS(self.vars, hess, self.cost, self.gradient, self.N)
        elif self.method == 'Newton LS':
            print(self.cost(self.vars))
            for i in range(10):
                self.vars = newton_iter_selection(self.vars, grad, hess, self.N, self.cost, 'on')
                grad = self.gradient(self.vars)
                hess = self.hessian(self.vars)
            print(self.cost(self.vars))
        elif self.method == 'Newton':
            for i in range(10):
                self.vars = newton_iter_selection(self.vars, grad, hess, self.N, self.cost, 'off')
                grad = self.gradient(self.vars)
                hess = self.hessian(self.vars)
                # self.vars = newton_iter(self.vars, grad, hess)
        elif self.method == 'Gradient':
            self.vars = gradient_search(self.vars, self.cost, self.gradient)
        elif self.method == 'Built-in optimizer':
            print(self.cost(self.vars))
            # result = minimize(fun=self.cost, x0=vars_select, method='trust-ncg', jac=self.gradient, hess=self.hessian, options = {'maxiter': 10})
            result = minimize(self.cost, self.vars, method='Nelder-Mead', options = {'maxiter': 30})
            self.vars = result.x
            print(self.cost(self.vars))
        else:
            print('Optimization method ' + self.method + ' non recognize')

    def last_covariance(self):
        self.ekf.x = self.vars[0:7]
        self.ekf.P = self.P0

        for i in range(self.N):
            self.ekf.F = self.dfdx(self.ekf.x)
            self.ekf.predict(fx=self.m.f)
            self.ekf.update(z=self.y[i+1], HJacobian=self.dh, Hx=self.o.h)
            if i == 0:
                P0 = np.copy(self.ekf.P)
        return self.ekf.P, P0

    def tuning_MHE(self, real_x, real_beta, step):
        self.real_x = np.ones((1+self.N)*7)
        for i in range(0, self.N+1):
            for j in range(3):
                self.real_x[i*7+j] = real_x[step-self.N-1 + i][0][j]
            for j in range(3):
                self.real_x[i*7+j+3] = real_x[step-self.N-1 + i][1][j]
            self.real_x[i*7 + 6] = real_beta[i]
        coeffs = []
        for i in range(3): coeffs.append(self.measurement_pen[i])
        for i in range(7): coeffs.append(self.model_pen[i])

        tuning = minimize(self.tuning_cost, coeffs, method='Nelder-Mead', options = {'maxiter': 200} )
        tuning.x = np.abs(tuning.x)
        for i in range(3):
            self.reg1[i] = tuning.x[i]
            self.measurement_pen[i] = tuning.x[i]
        for i in range(7):
            self.reg2[i] = tuning.x[i+3]
            self.model_pen[i] = tuning.x[i+3]
        return tuning.x

    def tuning_cost(self, coeffs):
        reg1 = np.ones(3)
        reg2 = np.ones(7)
        for i in range(3):
            reg1[i] = coeffs[i]
        for i in range(7):
            reg2[i] = coeffs[i+3]

        h_i = []
        g_i = np.zeros((self.N + 1) * 7)

        self.ekf.P = np.copy(self.P0)
        self.ekf.x = self.real_x[0:7]

        for i in range(0, self.N):
            # self.ekf.x = var[i*7:(i+1)*7]
            self.ekf.F = self.dfdx(self.ekf.x)
            self.ekf.predict(fx=self.m.f)
            self.ekf.update(z=self.y[i + 1], HJacobian=self.dh, Hx=self.o.h)
            g_i[i * 7:(i + 1) * 7] = self.ekf.x
            h_i.append(self.o.h(self.real_x[i * 7:i * 7 + 3], 'off'))
        h_i.append(self.o.h(self.real_x[self.N * 7:self.N * 7 + 3], 'off'))

        J = 0
        for i in range(self.N + 1):
            J = J + 0.5 * LA.norm(np.multiply(reg1, self.y[i] - h_i[i]))**2

        for i in range(self.N):
            J = J + 0.5 * self.pen * LA.norm(
                np.multiply(reg2, self.real_x[(i + 1) * 7:(i + 2) * 7] - g_i[i * 7:(i + 1) * 7])) ** 2
        for i in range(10):
            J = J + np.abs(10/coeffs[i])
        return J
