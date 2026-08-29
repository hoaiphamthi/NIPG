import numpy as np

# =============================================================================
# 1. SHARED NON-SMOOTH FUNCTIONS & PROXIMAL OPERATORS
# =============================================================================

def fun_g_row(X, lambda_row):
    return lambda_row * np.sum(np.linalg.norm(X, axis=1))

def fun_g_col(X, lambda_col):
    return lambda_col * np.sum(np.linalg.norm(X, axis=0))

def fun_g(X, lambda_row, lambda_col, mu=0.01):
    # Elastic Net penalty (Overlapping Group Lasso + squared Frobenius norm)
    # Added 'mu' to synchronize with both IQP and CUR requirements
    return fun_g_row(X, lambda_row) + fun_g_col(X, lambda_col) + (mu / 2) * (np.linalg.norm(X, 'fro') ** 2)

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


# =============================================================================
# 2. IQP PROBLEM (Indefinite Quadratic)
# =============================================================================

# Renamed to fun_grad_f_IQP to avoid conflicts
def fun_grad_f_IQP(A, B, X):
    """ f(X) = 0.5 * tr(X^T A X) + tr(B^T X) """
    f_val = 0.5 * np.trace(X.T @ A @ X) + np.trace(B.T @ X)
    grad_f = A @ X + B
    return f_val, grad_f

# Renamed to fun_F_IQP
def fun_F_IQP(A, B, X, lambda_row, lambda_col, mu=0.01):
    """ Used for logging history and plotting convergence graphs """
    r, _ = fun_grad_f_IQP(A, B, X)
    s = fun_g(X, lambda_row, lambda_col, mu)
    return r + s


# =============================================================================
# 3. CUR APPROXIMATION PROBLEM
# =============================================================================

# Renamed to fun_grad_f_CUR to avoid conflicts
def fun_grad_f_CUR(W, X):
    """ f(X) = 0.5 * ||W - WXW||_F^2 """
    prev = W - W @ X @ W
    fun = (1 / 2) * (np.linalg.norm(prev, 'fro')) ** 2
    gradf = -W.T @ prev @ W.T
    return fun, gradf

# Renamed to fun_F_CUR
def fun_F_CUR(W, X, lambda_row, lambda_col, mu=0.01):
    r, _ = fun_grad_f_CUR(W, X)
    s = fun_g(X, lambda_row, lambda_col, mu)
    return r + s