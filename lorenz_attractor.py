import numpy as np

class lorenz:
    def _init_(self):

        self.dt = 0.001 #default step size
        self.T = 100 # default final time
        self.N = self.T/self.dt # default number of time steps
        self.X = np.zeros((3,self.N)) # array to store X positions (x,y,z)

# main
dLz = lorenz()
