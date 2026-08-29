import numpy as np

# --- SMOOTH FUNCTION f(X) (Indefinite Quadratic) ---
def fun_grad_f(A, B, X):
    """ f(X) = 0.5 * tr(X^T A X) + tr(B^T X) """
    f_val = 0.5 * np.trace(X.T @ A @ X) + np.trace(B.T @ X)
    grad_f = A @ X + B
    return f_val, grad_f

# --- NON-SMOOTH FUNCTION g(X) (Overlapping Group Lasso) ---
def fun_g_row(X, lambda_row):
    return lambda_row * np.sum(np.linalg.norm(X, axis=1))

def fun_g_col(X, lambda_col):
    return lambda_col * np.sum(np.linalg.norm(X, axis=0))

def fun_g(X, lambda_row, lambda_col):
    return fun_g_row(X, lambda_row) + fun_g_col(X, lambda_col)

# --- TOTAL OBJECTIVE FUNCTION F(X) = f(X) + g(X) ---
def fun_F(A, B, X, lambda_row, lambda_col):
    """ Used for logging history and plotting convergence graphs """
    r, _ = fun_grad_f(A, B, X)
    s = fun_g(X, lambda_row, lambda_col)
    return r + s

# --- PROXIMAL OPERATOR (WITH BOX CONSTRAINTS ADDED) ---
def softThreshold_row(M, L, lamb, box_min=-10.0, box_max=10.0):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.maximum(0, 1 - (lamb / L) / norms)
    scale[np.isnan(scale)] = 0
    
    # Proximal step for Group Lasso
    X_prox = M * scale
    
    # BRAKING STEP: Projection onto Box Constraint to prevent overflow
    X_clipped = np.clip(X_prox, box_min, box_max)
    return X_clipped

def softThreshold_col(M, L, lamb, box_min=-10.0, box_max=10.0):
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.maximum(0, 1 - (lamb / L) / norms)
    scale[np.isnan(scale)] = 0
    
    # Proximal step for Group Lasso
    X_prox = M * scale
    
    # BRAKING STEP: Projection onto Box Constraint to prevent overflow
    X_clipped = np.clip(X_prox, box_min, box_max)
    return X_clipped