import numpy as np
# from scipy import linalg as LA
from numpy import linalg as LA
import sys
# import sympy

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

def newton_iter_selection(x0, grad, hess, cost, N):

    hessian = np.zeros(((N+1)*7, (N+1)*7))
    gradient = np.zeros((N+1)*7)
    x_zero = np.zeros((N+1)*7)

    for i in range(0, 2*N+1, 2):
        for j in range(0, 2*N+1, 2):
            hessian[(i//2)*7:(1+i//2)*7, (j//2)*7:(1+j//2)*7] = hess[i*7:(i+1)*7, j*7:(j+1)*7]
        gradient[(i//2)*7:(i//2+1)*7] = grad[i*7:(i+1)*7]
        x_zero[(i//2)*7:(i//2+1)*7] = x0[i*7:(i+1)*7]

    # print('max and min eigenvalues')
    # print(np.amax(LA.eigvals(hessian)))
    # print(np.amin(LA.eigvals(hessian)))
    # print('--')
    # print(np.amax(gradient))
    # print(np.amin(gradient))
    # print('--')
    # print(np.amax(hessian))
    # print(np.amin(hessian))

    # if LA.det(hessian)==0:
    #     print('eigenvalues')
    #     lambdas, V = np.linalg.eig(hessian.T)
    #     print('min')
    #     print(np.amin(lambdas))
    #     print(np.prod(lambdas))
    #     print('loc')
    #     for l in range(len(hessian)):
    #         if np.all(hessian[l] == hessian[lambdas == np.amin(lambdas)]):
    #             print('where true')
    #             print(l)
    #             print(l//7)
    #             print(l%3)
    #     print('sep')
    #     print(LA.det(hessian))
    #     print(LA.matrix_rank(hessian))
    #     sys.exit()

    # if not is_pos_def(hessian):
    #     print('negativve hessian')
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



    # mat= np.array(
    #     [
    #         [0, 1, 0, 0],
    #         [0, 0, 1, 0],
    #         [0, 1, 1, 0],
    #         [1, 0, 0, 1]
    #     ])

    # The linearly dependent row vectors
    # _, inds = sympy.Matrix(mat).T.rref()  # to check the rows you need to transpose!
    # print(inds)
    #
    # sys.exit()
    # lambdas, V = np.linalg.eig(hessian.T)
    # # The linearly dependent row vector
    # print(hessian[lambdas > 0, :])
    #
    # sys.exit()