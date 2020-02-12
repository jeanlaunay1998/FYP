import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class lorenz:
    def __init__(self):

        self.dt = 0.01 #default step size
        self.T = 100 # default final time
        self.N = int(self.T/self.dt) # default number of time steps

        self.X = np.zeros((self.N,3),dtype=float) # array to store X positions (x,y,z)

        self.K1 = np.zeros((1,3),dtype=float)
        self.K2 = np.zeros((1,3),dtype=float)
        self.K3 = np.zeros((1,3),dtype=float)
        self.K4 = np.zeros((1,3),dtype=float)


        self.X[0, 0] = -15.8;
        self.X[0, 1] = -17.48;
        self.X[0, 2] = 35.64;

        self.sigma = 10
        self.R = 24
        self.b = -8/3


    def step_update(self,step):
        self.K1[0,0] = self.dx(self.X,step)
        self.K1[0,1] = self.dy(self.X,step)
        self.K1[0,2] = self.dz(self.X,step)

        self.K2[0,0] = self.dx(self.X+self.K1/2,step)
        self.K2[0,1] = self.dy(self.X+self.K1/2,step)
        self.K2[0,2] = self.dz(self.X+self.K1/2,step)

        self.K3[0,0] = self.dx(self.X+self.K2/2,step)
        self.K3[0,1] = self.dy(self.X+self.K2/2,step)
        self.K3[0,2] = self.dz(self.X+self.K2/2,step)

        self.K4[0,0] = self.dx(self.X+self.K3,step)
        self.K4[0,1] = self.dy(self.X+self.K3,step)
        self.K4[0,2] = self.dz(self.X+self.K3,step)

        self.X[step+1,:] = self.X[step,:] + (1/6)*(self.K1+2*self.K2+2*self.K3+self.K4)

    def dx(self,x,step):
        return self.dt * self.sigma * (x[step,1] - x[step,0])

    def dy(self,x,step):
        return self.dt * (self.R*x[step,0] - x[step,0]*x[step,2] - x[step,1])

    def dz(self,x,step):
        return self.dt * (x[step,0]*x[step,1] + self.b*x[step,2])


# main

Lz = lorenz()

for step in range(Lz.N-1):
    Lz.step_update(step)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(Lz.X[:,0],Lz.X[:,1],Lz.X[:,2])
plt.show()
# ax.plot(1,2,3)
