import numpy as np
from numpy import linalg as LA
from scipy.optimize import minimize
from newton_method import newton_iter_selection
from newton_method import BFGS
from newton_method import gradient_search
import sys


class MHE_regularisation:
    def __init__(self, model, observer, horizon_length, measurement_lapse, model_pen, opt_method = 'Gradient', Q=[], R=[]):
        self.m = model  # copy direction of model to access all function of the class
        self.o = observer  # copy direction of observer to access all function of the class

        self.alpha = 0.1  # random size of the step (to be changed by Hessian of the matrix)
        self.N = horizon_length  # number of points in the horizon
        self.J = 0  # matrix to store cost function
        self.inter_steps = int(measurement_lapse/self.m.delta_t)

        self.x_apriori = np.zeros(7)
        self.vars = np.zeros(7)
        self.x_init = np.zeros(7)
        self.y = []
        self.beta = self.m.beta
        self.beta_apriori = self.m.beta

        self.x_solution = [0, 0, 0]
        self.mu1 = 1
        self.matrixR = []

        if R == []:
            self.reg1 = LA.inv(np.array([[50, 0, 0], [0, (1e-3), 0], [0, 0, (1e-3)]]))
        else:
            self.reg1 = LA.inv(np.power(R, 0.5))

        if Q == [] :
            self.R_mu = np.identity(7)
            for i in range(7): self.R_mu[i, i] = model_pen[i]
        else:
            self.R_mu = LA.inv(np.power(Q, 0.5))

        # -------------------------------------------------- #
        self.initial_coefs =[1,1,1,1,1]
        self.real_x = []
        self.real_beta = []
        # -------------------------------------------------- #
        self.method = opt_method


    def estimator_initilisation(self, step, y_measured):
        if step == self.N+1:
            self.y = np.array(y_measured)[step - self.N - 1:step, :]
            self.vars[0:3] = np.copy(self.m.Sk[step-1-self.N*self.inter_steps][0])
            self.vars[3:6] = np.copy(self.m.Sk[step-1-self.N*self.inter_steps][1])
            self.vars[6] = self.m.beta
            self.x_apriori = self.vars
            self.beta = self.m.beta  # Initial guess from model


    def slide_window(self, last_y):
        # slide horizon of measurements
        self.y[0:self.N] = self.y[1:self.N + 1]
        self.y[self.N] = last_y
        self.beta = self.vars[6]
        self.vars = self.m.f(self.vars)
        self.x_apriori = self.vars


    def cost(self, x):
        x_iplus1 = np.copy(x)
        J = 0.5*self.mu1*(LA.norm(np.matmul(self.R_mu, x_iplus1 - self.x_apriori))**2)
        for i in range(0, self.N+1):
            J = J + 0.5*pow(LA.norm(np.matmul(self.reg1, (self.y[i] - self.o.h(x_iplus1, 'off')))), 2)
            for j in range(self.inter_steps):
                x_iplus1 = self.m.f(x_iplus1, 'off')
        return J

    def estimation(self):
        if all(self.vars) == 0:
            print('MHE has failed, optimization not performed')
        else:
            grad = self.gradient(self.vars)
            hess = self.hessian(self.vars)
            x_0 = self.vars
            if self.method == 'BFGS':
                self.vars = BFGS(self.vars, hess, self.cost, self.gradient, self.N)
            elif self.method == 'Newton LS':
                self.vars = newton_iter_selection(self.vars, self.gradient, self.hessian, self.N, self.cost, 'on')
            elif self.method == 'Newton':
                for i in range(15):
                    self.vars = newton_iter_selection(self.vars, grad, hess, self.N, self.cost, 'off')
                    grad = self.gradient(self.vars)
                    hess = self.hessian(self.vars)
                    # self.vars = newton_iter(self.vars, grad, hess)
            elif self.method == 'Gradient':
                self.vars = gradient_search(self.vars, self.cost, self.gradient)
            elif self.method == 'Built-in optimizer':

                result = minimize(fun=self.cost, x0=self.vars, method='trust-ncg', jac=self.gradient, hess=self.hessian, options={'maxiter': 50})
                self.vars = result.x
            else:
                print('Optimization method ' + self.method + ' not recognize')
            if any(self.vars == x_0):
                print('opt failed')
                self.vars = np.zeros(len(self.vars))


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
        R_mu = np.power(self.R_mu, 2)
        x_i = np.copy(x)
        grad = np.matmul(R_mu, x_i - self.x_apriori)  # arrival cost derivative

        dfdx_i = []
        dhdx_i = []
        h_i = []

        for i in range(self.N):
            dfdx_i.append(self.dfdx(x_i))
            dhdx_i.append(self.dh(x_i))
            h_i.append(self.o.h(x_i, 'off'))
            x_i = self.m.f(x_i, 'off')

        dhdx_i.append(self.dh(x_i))
        h_i.append(self.o.h(x_i, 'off'))

        dhdx_i = np.array(dhdx_i)
        dfdx_i =np.array((dfdx_i))
        h_i = np.array(h_i)
        R1 = np.zeros((3, 3))
        for i in range(3):
            R1[i,:] = self.reg1[i,:]*self.reg1[i,i]

        grad = grad + np.matmul(np.transpose(dhdx_i[0]), np.matmul(R1, h_i[0]-self.y[0]))
        for i in range(1, self.N+1):
            dfdx_mult = dfdx_i[0]
            # apparently the derivative is better approximated this way (mathematically it does not make sens)
            for j in range(1, i):
                dfdx_mult = np.matmul(dfdx_i[j], dfdx_mult)

            A = np.matmul(dhdx_i[i], dfdx_mult) # dh(x_i)/dx_k-
            B = np.matmul(R1, h_i[i] - self.y[i])
            C = np.matmul(np.transpose(A), B)
            grad = grad + C

        # checking method
        # for l in range(len(x)):
        #     if (l//7)%2 == 0:
        #         eps = 0.1
        #         plus_eps = np.copy(x)
        #         plus_eps[l] = plus_eps[l]+eps
        #         minus_eps = np.copy(x)
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
        return grad

    def hessian(self, x):
        R_mu = np.power(self.R_mu, 2)
        H = self.mu1*R_mu
        x_i = np.copy(x)

        dfdx_i = []
        dhdx_i = []
        h_i = []
        for i in range(self.N):
            dfdx_i.append(self.dfdx(x_i))
            dhdx_i.append(self.dh(x_i))
            h_i.append(self.o.h(x_i, 'off'))
            x_i = self.m.f(x_i, 'off')
        dhdx_i.append(self.dh(x_i))
        h_i.append(self.o.h(x_i, 'off'))

        dhdx_i = np.array(dhdx_i)
        dfdx_i = np.array((dfdx_i))
        h_i = np.array(h_i)
        R1 = np.zeros((3, 3))
        for i in range(3):
            R1[i, :] = self.reg1[i, :] * self.reg1[i, i]
        H = H + np.matmul(np.transpose(dhdx_i[0]), np.matmul(R1, dhdx_i[0]))

        for i in range(1, self.N + 1):
            dfdx_mult = dfdx_i[0]
            # apparently the derivative is better approximated this way (mathematically it does not make sens)
            for j in range(1, i):
                dfdx_mult = np.matmul(dfdx_i[j], dfdx_mult)

            A = np.matmul(dhdx_i[i], dfdx_mult)  # dh(x_i)/dx_k-
            B = np.matmul(R1, A)
            C = np.matmul(np.transpose(A), B)
            H = H + C

        # checking method
        # for l in range(len(x)):
        #     if (l // 7) % 2 == 0:
        #         eps = 0.1
        #
        #         plus_eps = np.copy(x)
        #         plus_eps[l] = plus_eps[l]+eps
        #         minus_eps = np.copy(x)
        #         minus_eps[l] = minus_eps[l] - eps
        #
        #         A = self.gradient(plus_eps)
        #         B = self.gradient(minus_eps)
        #         derivative = (A-B)/(2*eps)
        #         # H[l, :] = derivative
        #         print(l, ': max diff (%): ', 100*np.amax(np.abs(np.divide(H[l,: ] - derivative, H[l,: ], out=np.zeros_like(H[l,: ]), where=H[l,: ]!=0))))
        #         print('numerical', derivative)
        #         print('analytical', H[l, :])
        #         print('----')
        # sys.exit()
        return H

    def last_state(self):
        x = np.copy(self.vars)
        for i in range(self.N):
            x = self.m.f(x, 'off')
        return x



    def initialisation2(self, y_measured, real_x, real_beta, step):
        self.real_x = real_x[step - self.N - 1]
        self.real_beta = real_beta[step - self.N - 2]
        self.initial_coefs = [1, 1, 1, 1, 1]

    def real_cost(self, coefficients):
        R = [0, 0, 0]
        mu1, R[0], R[1], R[2], mu2 = coefficients

        x_iplus1 = np.copy(self.real_x)
        x_inverse = 1 / self.x_apriori
        J = mu1 * (LA.norm((x_iplus1 - self.x_apriori) * x_inverse) ** 2) + mu2*(self.beta_apriori - self.real_beta) ** 2

        for i in range(0, self.N + 1):
            self.matrixR = np.array(
                [[R[0] / self.y[i, 0], 0, 0], [0, R[1] / self.y[i, 1], 0], [0, 0, R[2] / self.y[i, 2]]])
            # self.matrixR = [[1,0,0],[0,1,0],[0,0,1]]
            J = J + pow(LA.norm(np.matmul(self.matrixR, (self.y[i] - self.o.h(x_iplus1[0], 'off')))), 2)

            # note that since the model perform steps of 0.01 secs the step update needs to be perform 1/0.01 times to
            # obtain the value at same measurement time
            for j in range(self.inter_steps):
                x_iplus1[0], x_iplus1[1], a, beta_i = self.m.f(x_iplus1[0], x_iplus1[1], beta_i, 'off')

        J = J + (1/mu1) + (1/mu2) + (1/R[0]) + (1/R[1]) + (1/R[2])
        return J

    def search_coeffs(self):
        res = minimize(self.real_cost, self.initial_coefs, method='Nelder-Mead', tol=1e-6)
        self.mu1 = res.x[0]
        self.R[0] = res.x[1]
        self.R[1] = res.x[2]
        self.R[2] = res.x[3]

        print(res.x)