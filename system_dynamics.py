import numpy as np
from numpy import linalg as LA
from scipy.integrate import odeint

class dynamics:
    def __init__(self,ho,vo,gamma_o):

        # earth constants
        self.G = 6.673e-11
        self.M = 5.972*pow(10,24)
        self.R = 6371e3

        # space debris constants
        self.m = 15.8
        self.D = 0.58 # 3 meters of diameter debris
        self.A = np.pi*pow(self.D/2,2)

        # initialisation of state space
        # self.x = np.zeros((3,3))
        self.a = [0,0,0]
        self.v = [vo * np.cos(gamma_o * np.pi / 180), 0, vo * np.sin(gamma_o * np.pi / 180)]
        self.r = [0, 0, ho + self.R] # initial height #
        self.x = [[self.a, self.v, self.r]]

        # plotted variables
        self.h = [LA.norm(self.r)- self.R]
        self.beta = [self.ballistic_coef(self.v, self.r)]
        self.rho = [self.density_h(self.r)]

        # Runge Kutta parameters
        self.delta_t = 0.01



    def temp(self, r):
        height = LA.norm(r)-self.R
        if height<11000:
            ho = 0
            To = 288.15
            c = -0.0065
        elif height>= 11000 and height<20000:
            ho = 11000
            To = 216.65
            c = 0
        elif height>=20000 and height<32000:
            ho = 20000
            To = 216.5
            c = 0.001
        elif height>=32000 and height<47000:
            ho = 32000
            To = 228.65
            c = 0.0028
        elif height>=47000 and height<51000:
            ho = 47000
            To = 270.65
            c = 0
        elif height>=51000 and height<71000:
            ho = 51000
            To = 270.65
            c = -0.0028
        else:
            ho = 71000
            To = 214.65
            c = -0.002
        T = To + c * (height-ho)
        return T


    def visc(self,r):
        To = 291.15
        C = 130
        mu = 18.27e-6 * (To + C)*pow(self.temp(r)/To,3/2)/(self.temp(r) + C)
        return mu


    def density_h(self,r):
        height = LA.norm(r) - self.R

        if height<9144:
            c1 = 1.227
            c2 = 1.093e-4
            rho = c1*np.exp(-c2*height)
        else:
            c1 = 1.754
            c2 = 1.490e-4
            rho = c1*np.exp(-c2*height)
        return rho


    def reynolds(self, v, r):
        Re = self.density_h(r)*LA.norm(v)*self.D/self.visc(r)
        return Re


    def drag_coef(self, v, r):
        Re = self.reynolds(v,r)
        Cd = (24/Re) + 2.6*(Re/5)/(1 + pow(Re/5, 152)) + 0.411*pow(Re/263000, -7.94)/(1 + pow(Re/263000, -8)) + pow(Re, 0.8)/461000
        return Cd


    def ballistic_coef(self, v, r):

        beta =  self.m/(self.A*self.drag_coef(v, r))
        return beta


    def acceleration(self, v, r):
        acc = -np.multiply((self.G*self.M)/pow(LA.norm(r),3), r) - np.multiply(self.density_h(r)*LA.norm(v)/(2*self.ballistic_coef(v, r)), v)
        return list(acc)

    def dx(self, v, r):
        return self.acceleration(v, r), v

    def step_update(self, v, r):

        K1 = np.multiply(self.delta_t, self.dx(v, r))
        K2 = np.multiply(self.delta_t, self.dx(v+K1[0, :]/2, r+K1[1, :]/2))
        K3 = np.multiply(self.delta_t, self.dx(v+K2[0, :]/2, r+K2[1, :]/2))
        K4 = np.multiply(self.delta_t, self.dx(v+K3[0, :], r+K3[1, :]))

        self.a = self.acceleration(d.v, d.r)
        self.v, self.r = [self.v, self.r] + (1/6)*(K1+2*K2+2*K3+K4)
        self.v = self.v.tolist()
        self.r = self.r.tolist()
        self.x.append([self.a, self.v, self.r])

        self.h.append(LA.norm(self.r) - self.R)
        self.beta.append(self.ballistic_coef(self.v, self.r))
        self.rho.append(self.density_h(self.r))



# main
d = dynamics(80e3,6000,-5)
height = 80e3
t_lim = 240
t = 0
while height>500 and t<t_lim:
    d.step_update(d.v, d.r)
    height = d.h[len(d.h)-1]
    t = t + d.delta_t
    # print(t)


import matplotlib.pyplot as plt
time = np.linspace(0,t,len(d.h))
plt.figure(1)
plt.plot(time, d.beta, 'b', label='Ballistic coef')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.plot(time, d.h, 'b', label='Height (m)')
# plt.plot(time,sol[:,1],'r',label='Omega(t)')
plt.legend(loc='best')
plt.show()

plt.figure(3)
plt.plot(time, d.rho, 'b', label='Density')
# plt.plot(time,sol[:,1],'r',label='Omega(t)')
plt.legend(loc='best')
plt.show()

# print(d.x)
# d.step_update(d.v, d.r,1)
# print(np.array([d.a,d.v,d.r]))




