import numpy as np
from numpy import linalg as LA
from scipy.integrate import odeint

class dynamics:
    def __init__(self, ho, lat, long, vo, gamma_o, theta_o):
        # ho: initial height
        # lat: initial latitude
        # long: initial longitude
        # vo: initital velocity
        # gamma_o: initial Flight Path Angle (normal to the local horizontal plane)
        # theta_o: initial heading angle with respect to the north direction

        # change angles to radians
        long = np.pi*long/180
        lat = np.pi*lat/180
        gamma_o = np.pi*gamma_o/180
        theta_o = np.pi*theta_o/180

        # earth constants
        self.G = 6.673e-11
        self.M = 5.972*pow(10,24)
        self.R = 6371e3

        # space debris constants
        self.m = 15.8
        self.D = 0.58 # 3 meters of diameter debris
        self.A = np.pi*pow(self.D/2,2)

        # initialisation of state space
        v_NWU = [np.cos(theta_o)*np.cos(gamma_o), -np.sin(theta_o)*np.cos(gamma_o), np.sin(gamma_o)]
        v_NWU = [v_NWU[i]*vo for i in range(len(v_NWU))] # velocity in North West Up coordinates
        # transform to Earth Centred coordinates
        self.v = []
        self.v.append(-np.sin(lat)*np.cos(long)*v_NWU[0] + np.sin(long)*v_NWU[1] + np.cos(lat)*np.cos(long)*v_NWU[2])
        self.v.append(-np.sin(lat)*np.sin(long)*v_NWU[0] - np.cos(long)*v_NWU[1] + np.cos(lat)*np.sin(long)*v_NWU[2])
        self.v.append(np.cos(lat)*v_NWU[0] + np.sin(lat)*v_NWU[2])

        self.r = [(ho + self.R)*np.cos(lat)*np.cos(long), (ho + self.R)*np.cos(lat)*np.sin(long), (ho + self.R)*np.sin(lat)] # initial height #

        self.a = [0, 0, 0]
        self.x = [[self.a, self.v, self.r]]

        # plotted variables
        self.h = [ho]
        self.beta_o = self.m/(self.A*self.drag_coef(self.v, self.r))
        # white noise accounting for not modeled physics
        self.delta_o = np.random.normal(0, pow(0.01*self.beta_o, 2), size=1)
        self.a_res = 0 # np.random.normal(0, pow(0.01*LA.norm(self.a), 2), size=1)

        self.beta = [self.beta_o + self.delta_o]

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
        return beta + self.delta_o


    def acceleration(self, v, r):
        acc = -np.multiply((self.G*self.M)/pow(LA.norm(r),3), r) - np.multiply(self.density_h(r)*LA.norm(v)/(2*self.ballistic_coef(v, r)), v)
        return list(acc + self.a_res)

    def dx(self, v, r):
        return self.acceleration(v, r), v

    def step_update(self, v, r):

        K1 = np.multiply(self.delta_t, self.dx(v, r))
        K2 = np.multiply(self.delta_t, self.dx(v+K1[0, :]/2, r+K1[1, :]/2))
        K3 = np.multiply(self.delta_t, self.dx(v+K2[0, :]/2, r+K2[1, :]/2))
        K4 = np.multiply(self.delta_t, self.dx(v+K3[0, :], r+K3[1, :]))

        self.a = self.acceleration(self.v, self.r)
        self.v, self.r = [self.v, self.r] + (1/6)*(K1+2*K2+2*K3+K4)
        self.v = self.v.tolist()
        self.r = self.r.tolist()
        self.x.append([self.a, self.v, self.r])

        self.h.append(LA.norm(self.r) - self.R)
        self.beta.append(self.ballistic_coef(self.v, self.r))






