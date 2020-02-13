from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def pend(y,t):
    theta, omega = y
    dydt = [omega, -0.25*omega - 5.0*np.sin(theta)]
    return dydt

y0 = [np.pi -0.1, 0]
t = np.linspace(0,10,101)
sol = odeint(pend,y0,t)
plt.plot(t,sol[:,0],'b',label='Theta(t)')
plt.plot(t,sol[:,1],'r',label='Omega(t)')
plt.legend(loc='best')
plt.show()

from numpy import linalg as LA

def density_h(r):
    height = LA.norm(r)
    if height<9144:
        c1 = 1.227
        c2 = 1.093e-4
        rho = c1*np.exp(-c2*height)
    else:
        c1 = 1.754
        c2 = 1.490e-4
        rho = c1*np.exp(-c2*height)
    return rho

print(density_h(0))