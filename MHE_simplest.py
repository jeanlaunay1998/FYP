import numpy as np
from numpy import linalg as LA



class MHE:
    def __init__(self, model, observer):
        self.alpha = 0.1  # random size of the step (to be changed by Hessian of the matrix)
        self.N = 10  # number of points in the horizon
        self.J = 0  # matrix to store cost function

        self.m = model  # copy direction of model to access all function of the class
        self.o = observer  # copy direction of observer to access all function of the class
        self.x_apriori = []
        self.x_init = []
        self.y = []
        self.a = []
        self.beta = []
        self.grad = []
        self.x_solution = np.zeros((2, 3))

    def initialisation(self, y_measured, step):
        if step == self.N:
            self.y = np.array(y_measured)[step-self.N:step, :]
            self.x_apriori = np.array(self.m.Sk[step-self.N])
            self.x_init = self.x_apriori

            self.beta = self.m.beta # for the moment this works since beta is constant IT WILL BE CHANGE!)
            self.a = np.array(self.m.acceleration(self.x_apriori[0], self.x_apriori[1], self.beta)) # change withh acceleration function

        if step > self.N:
            self.y = np.array(y_measured)[step - self.N:step, :]#

            self.a = np.array(self.m.acceleration(self.x_solution[0], self.x_solution[1], self.beta))
            self.x_apriori[0], self.x_apriori[1], self.a, self.beta = self.m.f(self.x_solution[0], self.x_solution[1], self.a, self.beta)
            self.x_init = self.x_apriori

            self.beta = self.m.beta

    def cost_function(self, x):
        if len(x) != len(self.x_apriori):
            a = np.zeros((2,3))
            for i in range(len(x)):
                a[int(i/3), i%3] = x[i]
            x = np.array(a)

        self.J = 0.01*pow(LA.norm(x - self.x_apriori), 2) + pow(LA.norm(self.y[0] - self.o.h(x[0])), 2)
        x_iplus1 = x
        for i in range(self.N):
            x_iplus1[0], x_iplus1[1], self.a, self.beta = self.m.f(x_iplus1[0], x_iplus1[1], self.a, self.beta)
            self.J = self.J + pow(LA.norm(self.y[i] - self.o.h(x_iplus1[0])), 2)
        return self.J


    def density_constants(self, height):
        height = height - self.m.R
        if height < 9144:
            c1 = 1.227
            c2 = 1.093e-4
        else:
            c1 = 1.754
            c2 = 1.490e-4
        return [c1, c2]

# All functions below are used to compute the gradient of the cost function (an error exists in the derivation since the
    # real value is not working)
    def dacc_dr(self, r, v):
        norm_r = LA.norm(r)
        norm_v = LA.norm(v)

        # For legibility of the gradient constant terms across gradient are previously defined
        constant1 = self.m.G*self.m.M*pow(norm_r, -3)
        c1, c2 = self.density_constants(norm_r)
        constant2 = norm_v*c2*self.m.density_h(r)/(2*self.beta*norm_r)

        dA = constant1 * np.array([[-1 + pow(norm_r, -2)*r[0]*r[0], pow(norm_r, -2)*r[0]*r[1], pow(norm_r, -2)*r[0]*r[2]],
              [pow(norm_r, -2)*r[1]*r[0], -1 + pow(norm_r, -2)*r[1]*r[1], pow(norm_r, -2)*r[1]*r[2]],
              [pow(norm_r, -2)*r[2]*r[0], pow(norm_r, -2)*r[2]*r[1], -1 + pow(norm_r, -2)*r[2]*r[2]]])

        dB = np.array([[constant2[0]*v[0]*r[0], constant2[0]*v[0]*r[1], constant2[0]*v[0]*r[2]],
              [constant2[0]*v[1]*r[0], constant2[0]*v[1]*r[1], constant2[0]*v[1]*r[2]],
              [constant2[0]*v[2]*r[0], constant2[0]*v[2]*r[1], constant2[0]*v[2]*r[2]]])
        dadr = dA + dB
        return dadr

    def dacc_dv(self, r, v):
        norm_v = LA.norm(v)
        constant1 = -(self.m.density_h(r)/self.beta)
        dadv = np.array([[norm_v + pow(norm_v, -1)*v[0]*v[0], pow(norm_v, -1)*v[0]*v[1], pow(norm_v, -1)*v[0]*v[2]],
                        [pow(norm_v, -1)*v[1]*v[0], norm_v + pow(norm_v, -1)*v[1]*v[1], pow(norm_v, -1)*v[1]*v[2]],
                        [pow(norm_v, -1)*v[2]*v[0], pow(norm_v, -1)*v[2]*v[1], norm_v + pow(norm_v, -1)*v[2]*v[2]]])
        dadv = constant1*dadv
        return dadv

    def df(self, x):
        # x: point at which the derivative is evaluated
        r = x[0]
        v = x[1]

        # compute acceleration derivatives
        dadr = self.dacc_dr(r, v)
        dadv = self.dacc_dv(r, v)
        # total derivative
        dfdx = []
        for i in range(3):
            dfdx.append((1 + 0.5*pow(self.m.delta_t, 2)*dadr[i]).tolist() + (self.m.delta_t + 0.5*pow(self.m.delta_t, 2)*dadv[i]).tolist())
        for i in range(3):
            dfdx.append((self.m.delta_t*dadr[i]).tolist() + (1 + self.m.delta_t*dadr[i]).tolist())
        # for i in range(4): dfdx[i] = dfdx[i].tolist()
        return np.array(dfdx)

    def dh(self, x):
        # x: point at which the derivative is evaluated
        r = self.o.position_transform(x[0])

        norm_r = LA.norm(r)
        dhdx = [[r[0]/norm_r, r[1]/norm_r, r[2]/norm_r, 0, 0, 0]]
        constant1 = 1/(np.sqrt(1-pow(r[2]/norm_r, 2)))
        constant2 = -constant1*r[2]*pow(norm_r, -3)
        dhdx.append([constant2*r[0], constant2*r[1], constant1*(pow(norm_r, -1) + constant2*r[2]), 0, 0, 0])
        constant3 = 1/((1 + pow(r[1]/r[0], 2))*r[0])
        dhdx.append([constant3*r[1]/r[0], -constant3, 0, 0, 0, 0])
        dhdx = np.matmul(dhdx, self.o.T)
        return dhdx


    def gradient(self, x_o):
        a = 2*(x_o - self.x_apriori)
        self.grad = []
        for i in range(6): self.grad.append(a[int(i/3), i%3])
        self.grad = np.array(self.grad)

        dfdx_i = []
        dhdx_i = []
        h_i = []
        x_i = self.x_init
        beta_i = self.beta
        a_i = self.a

        for i in range(self.N):
            dfdx_i.append(self.df(x_i))
            dhdx_i.append(self.dh(x_i))
            h_i.append(self.o.h(x_i[0]))
            x_i[0], x_i[1], a_i, beta_i = self.m.f(x_i[0], x_i[1], a_i, beta_i)

        dhdx_i = np.array(dhdx_i)
        h_i = np.array(h_i)

        for i in range(self.N):
            dfdx_mult = dfdx_i[0]
            for j in range(i):
                dfdx_mult = np.matmul(dfdx_mult, dfdx_i[j+1])
            A = np.matmul(np.transpose(dfdx_mult), np.transpose(dhdx_i[i]))
            B = np.transpose(self.y[i] - h_i[i])
            C = 2*np.matmul(A, B)
            self.grad = self.grad + C

    def step_optimization(self, y, e, step):
        self.initialisation(y, step)
        for i in range(100):
            print(self.cost_function(self.x_init))
            self.gradient(self.x_init)
            self.x_init[0] = self.x_init[0] - self.alpha*self.grad[0:3]
            self.x_init[1] = self.x_init[1] - self.alpha*self.grad[3:6]



