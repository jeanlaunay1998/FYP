import numpy as np
from numpy import linalg as LA



class MHE:
    def __init__(self, model, observer, measurement_lapse):
        self.m = model  # copy direction of model to access all function of the class
        self.o = observer  # copy direction of observer to access all function of the class

        self.alpha = 0.1  # random size of the step (to be changed by Hessian of the matrix)
        self.N = 10  # number of points in the horizon
        self.J = 0  # matrix to store cost function
        self.inter_steps = int(measurement_lapse/self.m.delta_t)

        self.x_apriori = []
        self.x_init = []
        self.y = []
        self.a = []
        self.beta = []

        self.x_solution = np.zeros((2, 3))
        self.mu = 0.01

    def initialisation(self, y_measured, step):
        if step == self.N+1:
            self.y = np.array(y_measured)[step - self.N - 1:step, :]
            self.x_apriori = np.array(self.m.Sk[len(self.m.Sk)-1-self.N*self.inter_steps])  # there is N-1 intervals over the horizon
            self.x_init = np.array(self.m.Sk[len(self.m.Sk)-1-self.N*self.inter_steps])

            self.beta = self.m.beta  # for the moment this works since beta is constant IT WILL BE CHANGE!)
            self.a = np.array(self.m.acceleration(self.x_apriori[0], self.x_apriori[1], self.beta))  # change with acceleration function

        if step > self.N+1:
            self.y = np.array(y_measured)[step - self.N -1:step, :]
            self.a = np.array(self.m.acceleration(self.x_solution[0], self.x_solution[1], self.beta))
            self.x_apriori[0], self.x_apriori[1], self.a, self.beta = self.m.f(self.x_solution[0], self.x_solution[1], self.a, self.beta, 'off')
            self.x_init = np.array(self.x_apriori)

            self.beta = self.m.beta

    def cost_function(self, x):

        # built in optimisation function delivers the input in shape (6,1) when shape (2,3) is required
        if len(x) != len(self.x_apriori):
            a = np.zeros((2,3))
            for i in range(len(x)):
                a[int(i/3), i%3] = x[i]
            x = np.array(a)
        J = self.mu*pow(LA.norm(x - self.x_apriori), 2)
        x_iplus1 = np.copy(x)
        self.a = np.array(self.m.acceleration(x_iplus1[0], x_iplus1[1], self.beta))

        # print('y')
        for i in range(0, self.N+1):

            matrixR = np.array([[1/self.y[i, 0], 0, 0], [0, 1000/self.y[i, 1], 0], [0, 0, 100/self.y[i,2]]])
            # matrixR = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            J = J + pow(LA.norm(np.matmul(matrixR, (self.y[i] - self.o.h(x_iplus1[0], 'off')))), 2)

            # note that since the model perform steps of 0.01 secs the step update needs to be perform 1/0.01 times to
            # obtain the value at same measurement time
            for j in range(self.inter_steps):
                x_iplus1[0], x_iplus1[1], self.a, self.beta = self.m.f(x_iplus1[0], x_iplus1[1], self.a, self.beta, 'off')

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
        constant1 = -(self.m.density_h(r)/(2*self.beta))
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
        dfdx = np.zeros((6, 6))

        # d(r_k+1)/dr
        for i in range(3):
            for j in range(3):
                dfdx[i, j] = 0.5*pow(self.m.delta_t, 2)*dadr[i, j]
        for i in range(3):
            dfdx[i, i] = 1 + dfdx[i, i]

        # d(r_k+1)/dv
        for i in range(3):
            for j in range(3):
                dfdx[i, j+3] = 0.5*pow(self.m.delta_t, 2)*dadv[i, j]
        for i in range(3):
            dfdx[i, i+3] = self.m.delta_t + dfdx[i, i+3]

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

        return dfdx


    def dh(self, x):
        # x: point at which the derivative is evaluated
        r = self.o.position_transform(x[0])

        norm_r = LA.norm(r)
        dhdx = np.zeros((3,6))
        dhdx[0, range(3)] = np.matmul(r, self.o.transform_M)/norm_r

        constant1 = 1/(np.sqrt(1-pow(r[2]/norm_r, 2)))
        constant2 = -constant1*r[2]*pow(norm_r, -3)
        dhdx[1, range(3)] = np.matmul([constant2*r[0], constant2*r[1], constant1*(pow(norm_r, -1) + constant2*r[2])], self.o.transform_M)

        constant3 = 1/((1 + pow(r[1]/r[0], 2))*r[0])
        dhdx[2, range(3)] = np.matmul([constant3*r[1]/r[0], -constant3, 0], self.o.transform_M)

        return dhdx


    def gradient(self, x_i):

        grad = np.zeros(6)

        if len(x_i) != len(self.x_apriori):
            a = x_i
            x_i = []
            x_i = np.zeros((2,3))
            for i in range(6): x_i[int(i / 3), i % 3] = a[i]
        a = self.mu*np.array(x_i - self.x_apriori)
        for i in range(6): grad[i] = a[int(i/3), i%3]
        dfdx_i = []
        dhdx_i = []
        h_i = []
        beta_i = self.beta
        a_i = self.m.acceleration(x_i[0], x_i[1], beta_i)

        for i in range(self.N):
            dfdx_i.append(self.df(x_i))
            dhdx_i.append(self.dh(x_i))
            h_i.append(self.o.h(x_i[0], 'off'))
            x_i[0], x_i[1], a_i, beta_i = self.m.f(x_i[0], x_i[1], a_i, beta_i, 'off') # side note (we are computing an extra term)

        dhdx_i.append(self.dh(x_i))
        h_i.append(self.o.h(x_i[0], 'off'))

        dhdx_i = np.array(dhdx_i)
        dfdx_i =np.array((dfdx_i))
        h_i = np.array(h_i)

        matrixR = np.array([[1 / self.y[0, 0], 0, 0], [0, 100 / self.y[0, 1], 0], [0, 0, 100 / self.y[0, 2]]])
        # matrixR = [[1,0,0],[0,1,0],[0,0,1]]
        grad = grad + np.matmul(np.transpose(dhdx_i[0]), np.matmul(matrixR, h_i[0]-self.y[0]))
        for i in range(1, self.N+1):
            dfdx_mult = dfdx_i[0]

            # apparently the derivative is better approximated this way (mathematically it does not make sens)
            for j in range(1, i):
                dfdx_mult = np.matmul(dfdx_i[j], dfdx_mult)

            A = np.matmul(dhdx_i[i], dfdx_mult) # dh(x_i)/dx_k-N
            # matrixR = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            matrixR = np.array([[1 / self.y[i, 0], 0, 0], [0, 100 / self.y[i, 1], 0], [0, 0, 100 / self.y[i, 2]]])
            B = np.matmul(matrixR, h_i[i] - self.y[i])
            C = np.matmul(np.transpose(A), B)
            grad = grad + C
        grad = 2*grad
        return grad

