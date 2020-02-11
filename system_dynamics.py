import numpy as np

class dynamics:
    def __init__(self):
        self.a = np.empty([3,1],dtype = float)  # acceleration matrix
        self.v = np.empty([3,1], dtype = float) # velocity matrix
        self.r = np.array([0,0,pow(6400,3)]) # initial height of 29 km

        # earth constants
        self.G = 9.807
        self.M = 5.972*pow(10,24)
        self.R = pow(6371,3)


# main
d = dynamics()
print(d.G)
