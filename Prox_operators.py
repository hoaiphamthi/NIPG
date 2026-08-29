"""min_{W} (1\2)|W-(X_k-(t_k)*grad f(X_k))| + (t_k*lambda_col) sum|W_i| + (t_k*lambda_row) sum |W_j| + (t_k*mu)|W|_F^2"""

import numpy as np
# Import the vectorized proximal operators and objective functions
from fun_gradobjec import  fun_g_col, fun_g, softThreshold_row, softThreshold_col

# =============================================================================
# 1. EXACT PROXIMAL OPERATOR
# =============================================================================

def ProxExact(X_k, gradf_X, t_k, lambda_row, lambda_col, maxIter, mu=0.01):
    # L_eff based on stepsize t_k
    L_eff = (1.0 + mu * t_k) / t_k
    
    # Calculate Z_eff directly using the mathematical formula
    Z_eff = (1.0 / (1.0 + mu * t_k)) * (X_k - t_k * gradf_X)
    
    X = np.copy(Z_eff)
    P = np.zeros_like(X)
    Q = np.zeros_like(X)
    i = 1
    
    while i <= maxIter:
        # Prox with respect to columns
        X_old = X
        P_old = P
        Q_old = Q
        Y = softThreshold_col(X_old + P_old, L_eff, lambda_col)
        P = v1 = X_old + P_old - Y

        # Prox with respect to rows
        X = softThreshold_row(Y + Q_old, L_eff, lambda_row)
        Q = Y + Q_old - X

        v = np.linalg.norm(X_old + P_old + Q_old - Z_eff)
        
        # FIXED: Added lambda_col to fun_g_col
        eps = fun_g_col(X, lambda_col) - fun_g_col(Y, lambda_col) - np.trace(np.dot(np.transpose(X-Y), L_eff*v1))
        
        if max(v, eps) <= 1.0e-12:
            Xtilde = X
            break
        i += 1

    return X, i


# =============================================================================
# 2. INEXACT PROXIMAL OPERATOR FOR FIXSTEP
# =============================================================================

def ProxInexFixStep(X_k, gradf_X, sigma, t_k, lambda_row, lambda_col, maxIter, mu=0.01):
    # L_eff based on stepsize t_k
    L_eff = (1.0 + mu * t_k) / t_k
    
    # Z_orig represents X_k - t_k * gradf_X (used in the stopping condition)
    Z_orig = X_k - t_k * gradf_X
    
    # Calculate Z_eff directly using the mathematical formula
    Z_eff = (1.0 / (1.0 + mu * t_k)) * Z_orig
    
    X = np.copy(Z_eff)
    P = np.zeros_like(X)
    Q = np.zeros_like(X)
    i = 1
    
    while i <= maxIter:
        # Prox with respect to columns
        X_old = X
        P_old = P
        Q_old = Q
        Y = softThreshold_col(X_old + P_old, L_eff, lambda_col)
        P = v1 = X_old + P_old - Y

        # Prox with respect to rows
        X = softThreshold_row(Y + Q_old, L_eff, lambda_row)
        Q = Y + Q_old - X

        v = X_old + P_old + Q_old - Z_eff
        
        # FIXED: Added lambda_col to fun_g_col
        eps = fun_g_col(X, lambda_col) - fun_g_col(Y, lambda_col) - np.trace(np.dot(np.transpose(X-Y), L_eff*v1))

        # Evaluate the inexactness condition
        if np.linalg.norm(v)**2 + 2*eps/L_eff <= sigma*np.linalg.norm(Z_orig - X)**2:
            Xtilde = X
            break
        i += 1

    return X, i, v, eps


# =============================================================================
# 3. INEXACT PROXIMAL OPERATOR FOR IPG-ELS
# =============================================================================

def ProxInexIPG_LS(X_k, gradf_X, gamma_1, gamma_2, tau, alpha, t_k, lambda_row, lambda_col, maxIter, mu=0.01):
    # Scale matrix Z and adjust Lipschitz constant to L_eff based on t_k
    L_eff = (1.0 + mu * t_k) / t_k
    
    # Calculate Z_eff directly using the mathematical formula
    Z_eff = (1.0 / (1.0 + mu * t_k)) * (X_k - t_k * gradf_X)
    
    X = np.copy(Z_eff)
    P = np.zeros_like(X)
    Q = np.zeros_like(X)
    i = 1
    
    while i <= maxIter:
        # Prox with respect to columns
        X_old = X
        P_old = P
        Q_old = Q
        
        Y = softThreshold_col(X_old + P_old, L_eff, lambda_col)
        P = v1 = X_old + P_old - Y

        # Prox with respect to rows
        X = softThreshold_row(Y + Q_old, L_eff, lambda_row)
        Q = Y + Q_old - X

        v = X_old + P_old + Q_old - Z_eff
        
        eps = fun_g_col(X, lambda_col) - fun_g_col(Y, lambda_col) - np.trace(np.dot(np.transpose(X-Y), L_eff*v1))
        
        aux = fun_g(X-v, lambda_row, lambda_col) - fun_g(X, lambda_row, lambda_col) - np.trace(np.dot(np.transpose(v), gradf_X)) + ((1+gamma_1)/2)*np.linalg.norm(v)**2 + (1+gamma_2)*eps

        if aux <= ((1-tau-alpha)/2)*np.linalg.norm(X_k-X)**2:
            Xtilde = X
            break
        i += 1

    return X, i, v, eps


# =============================================================================
# 4. INEXACT PROXIMAL OPERATOR FOR NIPGL
# =============================================================================

def ProxInexNIPGl(X_k, gradf_X, theta_1, theta_2, tau, t_k, lambda_row, lambda_col, maxIter, mu=0.01):
    # L_eff based on stepsize t_k
    L_eff = (1.0 + mu * t_k) / t_k
    
    # Calculate Z_eff directly using the mathematical formula from the manuscript
    Z_eff = (1.0 / (1.0 + mu * t_k)) * (X_k - t_k * gradf_X)
    
    # Initialize points z_0 = z, p_0 = 0, q_0 = 0
    X = np.copy(Z_eff) 
    P = np.zeros_like(X)
    Q = np.zeros_like(X)
    i = 1
    
    while i <= maxIter:
        # Prox with respect to columns
        X_old = X
        P_old = P
        Q_old = Q
        Y = softThreshold_col(X_old + P_old, L_eff, lambda_col)
        P = v1 = X_old + P_old - Y

        # Prox with respect to rows
        X = softThreshold_row(Y + Q_old, L_eff, lambda_row)
        Q = Y + Q_old - X

        v = X_old + P_old + Q_old - Z_eff
        
        eps = fun_g_col(X, lambda_col) - fun_g_col(Y, lambda_col) - np.trace(np.dot(np.transpose(X-Y), L_eff*v1))
        
        aux = fun_g(X-v, lambda_row, lambda_col) - fun_g(X, lambda_row, lambda_col) - np.trace(np.dot(np.transpose(v), gradf_X)) + (theta_1*L_eff)*np.linalg.norm(v)**2 + (1+theta_2)*eps

        # Inexact stopping condition: (tau * L) is replaced by (tau / t_k)
        if aux <= (tau / t_k) * np.linalg.norm(X_k - X)**2:
            Xtilde = X
            break
        i += 1

    return X, i, v, eps