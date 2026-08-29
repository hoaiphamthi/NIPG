import numpy as np

def fun_grad_f(W, X):
    prev = W - W @ X @ W
    fun = (1 / 2) * (np.linalg.norm(prev, 'fro')) ** 2
    gradf = -W.T @ prev @ W.T
    return fun, gradf

def fun_g_row(X):
    lamb_row = 0.01
    sum1 = 0
    for i in range(0, len(X)):
        sum1 += np.linalg.norm(X[i])
    fun = lamb_row * sum1
    return fun

def fun_g_col(X):
    lamb_col = 0.01
    sum2 = 0
    for i in range(0, len(X[0])):
        sum2 += np.linalg.norm(X[:, i])
    fun = lamb_col * sum2
    return fun

def fun_g(X, mu=0.01):
    # Add the Elastic penalty term (squared Frobenius norm)
    fun = fun_g_row(X) + fun_g_col(X) + (mu / 2) * (np.linalg.norm(X, 'fro') ** 2)
    return fun

def fun_F(W, X, mu=0.01):
    [r, _] = fun_grad_f(W, X)
    s = fun_g(X, mu)
    fun = r + s
    return fun