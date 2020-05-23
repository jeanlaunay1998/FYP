import numpy as np
# from scipy import linalg as LA
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

def newton_iter_selection(x, grad_fun, hess_fun, N, cost_fun, linesearch='off'):
    x0 = np.copy(x)
    for i in range(15):
        gradient = grad_fun(x0)
        hessian = hess_fun(x0)
        if LA.det(hessian) == 0:
            hessian = hessian + np.identity(len(hessian)) * 1e-7
            print('-----------------------------')
        p = np.matmul(LA.inv(hessian), gradient)
        if linesearch == 'on':
            alpha = line_Search(x0, cost_fun, p, gradient)
        else:
            alpha = 1
        x0 = x0 - alpha * p
    return x0

def BFGS(x0, B0, cost_fun, gradient, N):

    x_k = np.zeros((N + 1) * 7)
    B = np.zeros(((N + 1) * 7, (N + 1) * 7))

    for i in range(0, 2*N+1, 2):
        for j in range(0, 2*N+1, 2):
            B[(i//2)*7:(1+i//2)*7, (j//2)*7:(1+j//2)*7] = B0[i*7:(i+1)*7, j*7:(j+1)*7]
        x_k[(i//2)*7:(i//2+1)*7] = x0[i*7:(i+1)*7]
    grad = gradient(x_k)

    if LA.det(B) == 0:
        B = B + np.identity(len(B)) * 10e-7
        print('-----------------------------')

    for i in range(10):
        if i != 0:
            y_square = np.outer(y_k, np.transpose(y_k))
            ymultS = np.matmul(np.transpose(y_k), S_k)
            BmultS = np.matmul(B, S_k)
            SmultBmultS = np.matmul(np.transpose(S_k), BmultS)
            B = B + y_square[0]/ymultS - np.matmul(BmultS, np.transpose(BmultS))/SmultBmultS

        # if LA.det(B) == 0:
        #     B = B + np.identity(len(B)) * 10e-7
        #     print('-----------------------------')

        p = np.matmul(LA.inv(B), grad)
        alpha = line_Search(x_k, cost_fun, p, grad)

        S_k = - alpha * p
        x_k = x_k + S_k
        y_k = gradient(x_k) - grad
        grad = gradient(x_k)

    output = np.ones((2 * N + 1) * 7)
    for i in range(0, 2 * N + 1, 2):
        output[i * 7:(i + 1) * 7] = x_k[(i // 2) * 7:(i // 2 + 1) * 7]
    return output

def line_Search(x, cost_fun, p, gradient):
    # perform an Armijo line search
    # set alpha equal to maximum step
    alpha = 1
    sigma = 0.5
    c = 0.25
    x_kplus1 = x - alpha*p
    cost0 = cost_fun(x)
    while cost_fun(x_kplus1)-cost0 >= c*alpha*np.matmul(np.transpose(gradient), p):
        alpha = alpha*sigma
        x_kplus1 = x - alpha * p
        if alpha<1e-22:
            print('Failed search')
            alpha = 0
            break
    return alpha

def gradient_search(x0, cost_fun, gradient_fun):
    x = np.copy(x0)
    cost_before = cost_fun(x0)
    print('cost', cost_fun(x0))
    for i in range(20):
        p = gradient_fun(x)
        alpha = line_Search(x, cost_fun, p, p)
        x = x - alpha*p
    cost_after = cost_fun(x)
    print('cost', cost_after)
    if cost_after>=cost_before:
        return x0
    if cost_after<cost_before:
        return x


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)



# print('max and min eigenvalues')
    # lambdas = LA.eigvals(hessian)
    # print(lambdas)
    # print(np.prod(LA.eigvals(hessian)))
    # print(np.amax(lambdas)/np.amin(lambdas))
    # for l in range(len(hessian)):
    #     if np.all(hessian[l] == hessian[lambdas == np.amin(lambdas)]):
    #         print('where true')
    #         print(l)
    #         print(l//7)
    #         print(l%7)
    #         print(hessian[l,l])

    # if not is_pos_def(hessian):
    #     print('negativve hessian')
    # print(LA.eigvals(hessian))
    # sys.exit()