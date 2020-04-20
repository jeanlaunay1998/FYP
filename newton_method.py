import numpy as np
from numpy import linalg as LA
import sys


def newton_iter(x0, grad, hess):
    if LA.norm(grad) < 10e-4:
        print(x0)
        print('convergence criterion satisfied')
        sys.exit()

    if not is_pos_def(hess):
        print('negativve hessian')
        print(LA.eigvals(hess))
        # sys.exit()

    hess_inv = LA.inv(hess)
    output = x0 - np.matmul(hess_inv, grad)
    return output

def newton_iter_selection(x0, grad, hess, N):

    if LA.norm(grad) < 10e-4:
        print(x0)
        print('convergence criterion satisfied')
        sys.exit()

    hessian = np.zeros(((N+1)*7, (N+1)*7))
    gradient = np.zeros((N+1)*7)
    x_zero = np.zeros((N+1)*7)

    for i in range(0, 2*N+1, 2):
        for j in range(0, 2*N+1, 2):
            hessian[(i//2)*7:(1+i//2)*7, (j//2)*7:(1+j//2)*7] = hess[i*7:(i+1)*7, j*7:(j+1)*7]
        gradient[(i//2)*7:(i//2+1)*7] = grad[i*7:(i+1)*7]
        x_zero[(i//2)*7:(i//2+1)*7] = x0[i*7:(i+1)*7]

    if not is_pos_def(hessian):
        print('negativve hessian')
        # print(LA.eigvals(hessian))
        # sys.exit()

    hess_inv = LA.inv(hessian)
    x_zero = x_zero - np.matmul(hess_inv, gradient)

    output = np.ones((2*N+1)*7)
    for i in range(0, 2 * N + 1, 2):
        output[i * 7:(i + 1) * 7] = x_zero[(i // 2) * 7:(i // 2 + 1) * 7]
    return output


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)