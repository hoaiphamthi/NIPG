import numpy as np
import time
from fun_gradobjec import fun_grad_f_IQP, fun_F_IQP
from Prox_operators import ProxExact, ProxInexFixStep, ProxInexIPG_LS, ProxInexNIPGl


# =============================================================================
#HELPER FUNCTIONS
# =============================================================================

def get_gamma_err(k):
    if k == 0:
        return 1.0 
    return (0.5 * np.log(k+1)**5.7) / (k+1)**1.1

def LSExplicit(X, Y, f_X, prev, beta_init, A, B, theta):
    i = 0
    beta = beta_init
    Xplus = X + beta * (Y - X)
    [f_t, _] = fun_grad_f_IQP(A, B, Xplus)
    
    while f_t > f_X + beta * prev:
        beta *= theta
        Xplus = X + beta * (Y - X)
        [f_t, _] = fun_grad_f_IQP(A, B, Xplus)
        i += 1
    return beta, i, f_t, Xplus

def LSExplicit_exact(X, Y, f_X, prev, beta_init, A, B, theta):
    i = 0
    beta = beta_init
    Xplus = X + beta * (Y - X)
    [f_t, _] = fun_grad_f_IQP(A, B, Xplus)
    
    while f_t > f_X + beta * prev:
        beta *= theta
        Xplus = X + beta * (Y - X)
        [f_t, _] = fun_grad_f_IQP(A, B, Xplus)
        i += 1
    return beta, i, f_t, Xplus


# =============================================================================
# Main algorithms
# =============================================================================

def IPG_ELS(A, B, X0, f_X, gradf_X, gamma_1, gamma_2, theta, tau, alpha, t_k, kmax, prec, lambda_row, lambda_col, FOpt=None, mu=0.01):
    k = 0
    X = X0
    F_func = [fun_F_IQP(A, B, X, lambda_row, lambda_col, mu)]
    X_func = [np.copy(X0)] 
    i = 0
    ItLiSear = 0
    maxItInner = 100
    elapsed_times = [0]
    sun_it_time = 0
    
    while k <= kmax:
        start_time = time.time()
        
        if FOpt is not None and F_func[-1] <= FOpt:
            break
            
        [Xtilde, iprox, V, eps] = ProxInexIPG_LS(X, gradf_X, gamma_1, gamma_2, tau, alpha, t_k, lambda_row, lambda_col, maxItInner, mu)
        i = i + iprox
        
        Y = Xtilde - V
        prev = -np.sum(gradf_X * (X - Y)) + (tau/2) * (np.linalg.norm(X - Xtilde))**2 + (gamma_1/2) * np.linalg.norm(V)**2 + gamma_2 * eps
        
        [beta, ItLi, fun_Xplus, Xplus] = LSExplicit(X, Y, f_X, prev, 1, A, B, theta)
        ItLiSear += ItLi

        X = Xplus
        f_X = fun_Xplus
        [_, gradf_X] = fun_grad_f_IQP(A, B, X)
        k += 1
        end_time = time.time()
        
        F_func.append(fun_F_IQP(A, B, Xplus, lambda_row, lambda_col, mu))
        X_func.append(np.copy(X)) 
        sun_it_time += end_time - start_time
        elapsed_times.append(sun_it_time)

    return X, f_X, k, i, ItLiSear, F_func, elapsed_times, X_func


def IPG_Fixstep(A, B, X0, f_X, gradf_X, sigma, t_k, gamma, kmax, prec, lambda_row, lambda_col, FOpt, mu=0.01):
    k = 0
    X = X0
    F_func = [fun_F_IQP(A, B, X, lambda_row, lambda_col, mu)]
    X_func = [np.copy(X0)] 
    i = 0
    maxItInner = 100
    elapsed_times = [0]
    sun_it_time = 0
    
    while k <= kmax:
        start_time = time.time()
        
        [Xtilde, iprox, V, eps] = ProxInexFixStep(X, gradf_X, sigma, t_k, lambda_row, lambda_col, maxItInner, mu)
        i = i + iprox

        if fun_F_IQP(A, B, X, lambda_row, lambda_col, mu) <= FOpt:
            break
            
        X = X + gamma * (Xtilde - V - X)
        [f_X, gradf_X] = fun_grad_f_IQP(A, B, X)
        
        k += 1
        end_time = time.time()
        
        F_func.append(fun_F_IQP(A, B, X, lambda_row, lambda_col, mu))
        X_func.append(np.copy(X)) 
        sun_it_time += end_time - start_time
        elapsed_times.append(sun_it_time)

    return X, f_X, k, i, F_func, elapsed_times, X_func


def NIPGl(
    A, B, X0, f_X_initial, gradf_X_initial, lambda_row, lambda_col,
    kmax, c0, c1, theta_1, theta_2, tau, initial_step_size,
    option=1, maxItInner=100, FOpt=None, mu=0.01
):
    k = 0
    X = np.copy(X0) 
    f_X = f_X_initial
    gradf_X = np.copy(gradf_X_initial) 

    t_k = initial_step_size 
    t_prev_prev = initial_step_size 
    
    F_func = [fun_F_IQP(A, B, X, lambda_row, lambda_col, mu)]
    elapsed_times = [0]
    sun_it_time = 0
    total_inner_iterations = 0
    X_func = [np.copy(X0)] 
    step_sizes = [initial_step_size]
    
    while k <= kmax:
        if FOpt is not None and F_func[-1] <= FOpt:
            break
            
        start_time = time.time()
        
        X_k = np.copy(X) 
        gradf_X_k = np.copy(gradf_X) 
        t_current = t_k 

        [Xtilde, iprox, V, eps] = ProxInexNIPGl(X_k, gradf_X_k, theta_1, theta_2, tau, t_current, lambda_row, lambda_col, maxItInner, mu)
        
        total_inner_iterations += iprox
        X_new = Xtilde - V  
         
        [f_X_new, gradf_X_new] = fun_grad_f_IQP(A, B, X_new) 
        f_X = f_X_new 
        t_prev = t_current

        diff_X = X_k - X_new
        diff_grad = gradf_X_k - gradf_X_new
        
        norm_dx = np.sqrt(np.sum(diff_X ** 2))
        norm_d_grad = np.sqrt(np.sum(diff_grad ** 2))
        
        if norm_dx > 1e-12: 
            if option == 1:
                A_l = np.abs(np.sum(diff_grad * diff_X))
                if A_l > (c0 * (norm_dx**2))/t_k:
                    t_next = c1 * (norm_dx**2) / A_l
                    t_k = t_next 
                else:
                    gamma_k_ext = get_gamma_err(k)
                    gamma_prime = gamma_k_ext
                    if t_prev / t_prev_prev < 1:
                        gamma_prime = np.min([gamma_k_ext, np.sqrt(1 + t_prev / t_prev_prev) - 1])
                    t_k = (1 + gamma_prime) * t_prev
                    
            elif option == 2:
                A_l = norm_d_grad * norm_dx
                if A_l > (c0 * (norm_dx**2))/t_k:
                    t_next = c1 * (norm_dx**2) / A_l
                    t_k = t_next 
                else:
                    gamma_k_ext = get_gamma_err(k)
                    gamma_prime = gamma_k_ext
                    if t_prev / t_prev_prev < 1:
                        gamma_prime = np.min([gamma_k_ext, np.sqrt(1 + t_prev / t_prev_prev) - 1])
                    t_k = (1 + gamma_prime) * t_prev
            else:
                raise ValueError("The 'option' parameter must be 1 or 2")
        else:
            t_k = t_prev
            
        t_k = max(t_k, 1e-12)

        X = X_new
        gradf_X = gradf_X_new
        t_prev_prev = t_prev 
        k += 1
        
        step_sizes.append(t_k)
        end_time = time.time()
        
        F_func.append(fun_F_IQP(A, B, X, lambda_row, lambda_col, mu))
        sun_it_time += end_time - start_time
        elapsed_times.append(sun_it_time)
        X_func.append(np.copy(X))

    return X, f_X, k, total_inner_iterations, F_func, elapsed_times, X_func, step_sizes


def PG_ELS(A, B, X0, f_X, gradf_X, theta, t_k, kmax, prec, lambda_row, lambda_col, FOpt, mu=0.01):
    k = 0
    X = X0
    F_func = [fun_F_IQP(A, B, X, lambda_row, lambda_col, mu)]
    X_func = [np.copy(X0)] 
    i = 0
    ItLiSear = 0
    maxItInner = 100
    elapsed_times = [0]
    sun_it_time = 0
    
    while k <= kmax:
        start_time = time.time()
        
        [Xtilde, iprox] = ProxExact(X, gradf_X, t_k, lambda_row, lambda_col, maxItInner, mu)
        i = i + iprox
        
        if FOpt is not None and F_func[-1] <= FOpt:
            break
            
        Y = Xtilde
        prev = -np.sum(gradf_X * (X - Y)) + (1/2) * (np.linalg.norm(X - Xtilde))**2
        
        [beta, ItLi, fun_Xplus, Xplus] = LSExplicit_exact(X, Y, f_X, prev, 1, A, B, theta)
        ItLiSear += ItLi

        X = Xplus
        f_X = fun_Xplus
        [_, gradf_X] = fun_grad_f_IQP(A, B, X)
        k += 1
        end_time = time.time()
        
        F_func.append(fun_F_IQP(A, B, Xplus, lambda_row, lambda_col, mu))
        X_func.append(np.copy(X)) 
        sun_it_time += end_time - start_time
        elapsed_times.append(sun_it_time)

    return X, f_X, k, i, ItLiSear, F_func, elapsed_times, X_func