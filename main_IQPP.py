"""Indefinite Quadratic Optimization Problem with Group Lasso"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import csv
import gc
from fun_gradobjec import fun_grad_f_IQP, fun_F_IQP
from algos_iqp import IPG_ELS, PG_ELS, IPG_Fixstep, NIPGl

# --- SETUP DIRECTORIES FOR IQP ---
BASE_DIR = 'result/IQP'
IMG_DIR = os.path.join(BASE_DIR, 'images')
CSV_DIR = os.path.join(BASE_DIR, 'csv_results')
DATA_DIR = os.path.join(BASE_DIR, 'saved_data')

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_safe_filename(title_str):
    """Helper function to convert a plot title into a valid cross-platform filename."""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '$']
    safe_title = title_str
    for char in invalid_chars:
        safe_title = safe_title.replace(char, '')
    return safe_title.replace(' ', '_') + '.png'


# --- PLOTTING FUNCTIONS ---
def plot_F_vs_Time(algorithm_results, F_opt=None, problem_name="", title_details=""):
    plt.figure(figsize=(10, 8)) 
    
    markers = ['o', '*', '^', 's', 'D', 'v']
    
    for idx, (F_func, X_func, time_func, step_sizes, label, color) in enumerate(algorithm_results):
        F_to_plot = np.array(F_func)
        T_axis = np.array(time_func)
        
        current_marker = markers[idx % len(markers)]
        line_kwargs = {'label': label, 'color': color, 'marker': current_marker, 'markevery': max(1, len(T_axis)//20), 'markersize': 10, 'linewidth': 3}
        
        if F_opt is not None:
             F_to_plot = F_to_plot - F_opt
             F_to_plot[F_to_plot <= 1e-18] = 1e-18 
             plt.semilogy(T_axis, F_to_plot, **line_kwargs)
        else:
             plt.plot(T_axis, F_to_plot, **line_kwargs)

    title_str = f'{problem_name} - Convergence of F over Time' + title_details
    plt.title(title_str)
    plt.xlabel('Time (seconds)')
    plt.ylabel(r'$F(X^k) - F^*$ (Log Scale)' if F_opt is not None else r'$F(X^k)$')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    filepath = os.path.join(IMG_DIR, get_safe_filename(title_str))
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

def plot_StepSizes_vs_Iterations(algorithm_results, problem_name="", title_details=""):
    plt.figure(figsize=(10, 8))
    title_str = f'{problem_name} - Adaptive Step Size $t_k$ over Iterations' + title_details
    
    markers = ['o', '*', '^', 's', 'D', 'v']
    
    has_data = False
    for idx, (F_func, X_func, time_func, step_sizes, label, color) in enumerate(algorithm_results):
        if step_sizes is not None and len(step_sizes) > 1:
            iterations = np.arange(len(step_sizes))
            
            current_marker = markers[idx % len(markers)]
            plt.semilogy(iterations, step_sizes, label=label, color=color, marker=current_marker, markevery=max(1, len(step_sizes)//20), markersize=10, linewidth=3)
            has_data = True
            
    if has_data:
        plt.title(title_str)
        plt.xlabel('Outer Iterations (k)')
        plt.ylabel(r'Step Size $t_k$ (Log Scale)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        filepath = os.path.join(IMG_DIR, get_safe_filename(title_str))
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

def generate_indefinite_problem(n, m, neg_ratio=0.15,L_max=80, seed=42):
    """
    Generate data for the Indefinite Quadratic problem.
    """
    np.random.seed(seed)
    
    eigenvalues = np.random.uniform(0.1, L_max, n)
    num_negative = int(neg_ratio * n)
    eigenvalues[:num_negative] = np.random.uniform(-5, -0.1, num_negative)
    np.random.shuffle(eigenvalues)
    
    U_raw = np.random.randn(n, n)
    U, _ = np.linalg.qr(U_raw)
    A = U @ np.diag(eigenvalues) @ U.T
    B = np.random.randn(n, m)
    
    Lips = np.max(np.abs(eigenvalues))
    return A, B, Lips
    
dimensions = [
    (3000, 150),
    (500, 500),
    (250, 2500),
]

# ==========================================

active_dimensions = dimensions
csv_file_path = os.path.join(CSV_DIR, 'summary_results_all_dimensions_0.5_10_08.csv')

# --- RUN EXPERIMENTS ---
divider = "-" * 102
print(divider)
print(f'| {"Problem":<15} | {"Lips":>8} | {"Method":<12} | {"F(X^k)":>10} | {"Out-IT":>7} | {"In-IT":>7} | {"ELS-IT":>7} | {"Time(s)":>8} |')
print(divider)

all_results_table = []
lipschitz_values = [10, 30, 50, 80]

for i, (n, m) in enumerate(active_dimensions):
    for L_target in lipschitz_values: 
        name = f'n={n}, m={m}'
    
        A, B, Lips = generate_indefinite_problem(n, m, neg_ratio=0.15, L_max=L_target, seed=42 + n + m + int(L_target))   

        initial_t0 = 0.001
        X0 = np.zeros((n, m))
        mu = 0.0
        
        f_x, gradf_x = fun_grad_f_IQP(A, B, X0)
        
        prec = 1.0e-3
        kmax = 100
        lambda_row = lambda_col = 0.01
    
        gamma_1, gamma_2, theta, tau, alpha, t_k = 1.1, 1.1, 0.5, 0.8, 0.01, 1
        startFBinexLS2 = time.time()
        [X2, f_X2, k2, i2, ItLiSear2, F_func2, time_IT_LS2, X_func_c2] = IPG_ELS(
            A, B, X0, f_x, gradf_x, gamma_1, gamma_2, theta, tau, alpha, t_k, kmax, prec, lambda_row, lambda_col, mu=mu
        )
        endFBinexLS2 = time.time()
        time_2 = endFBinexLS2 - startFBinexLS2
    
        F_opt_val = fun_F_IQP(A, B, X2, lambda_row, lambda_col, mu)
        X_opt_val = X2
    
        print(f'| {name:<15} | {Lips:>8.4f} | {"IPG-ELS":<12} | {F_opt_val:>10.4f} | {k2:>7} | {i2:>7} | {ItLiSear2:>7} | {time_2:>8.2f} |')
        all_results_table.append({'Problem': name, 'Dims': f'{n}x{m}', 'Lips': Lips, 'Method': 'IPG-ELS', 'F(X^k)': F_opt_val, 'Out-IT': k2, 'In-IT': i2, 'ELS-IT': ItLiSear2, 'Time(s)': time_2})

        startFBLS = time.time()
        [XFB, f_XFB, kFB, iFB, ItLiSearFB, F_funcFB, time_IT_FB, X_func_FB] = PG_ELS(
            A, B, X0, f_x, gradf_x, theta, t_k, 2000, prec, lambda_row, lambda_col, F_opt_val, mu=mu
        )
        endFBLS = time.time()
        time_FB = endFBLS - startFBLS
        f_FB_val = fun_F_IQP(A, B, XFB, lambda_row, lambda_col, mu)
    
        print(f'| {"":<15} | {"":>8} | {"PG-ELS":<12} | {f_FB_val:>10.4f} | {kFB:>7} | {iFB:>7} | {ItLiSearFB:>7} | {time_FB:>8.2f} |')
        all_results_table.append({'Problem': name, 'Dims': f'{n}x{m}', 'Lips': Lips, 'Method': 'PG-ELS', 'F(X^k)': f_FB_val, 'Out-IT': kFB, 'In-IT': iFB, 'ELS-IT': ItLiSearFB, 'Time(s)': time_FB})

        sigma, t_k, gamma = 0.9, 1.0 / Lips, 1
        startFBinex = time.time()
        [Xinex, f_Xinex, kinex, iinex, F_funcinex, time_IT_FBinex, X_func_FBinex] = IPG_Fixstep(
            A, B, X0, f_x, gradf_x, sigma, t_k, gamma, 2000, prec, lambda_row, lambda_col, F_opt_val, mu=mu
        )
        endFBinex = time.time()
        time_inex = endFBinex - startFBinex
        f_inex_val = fun_F_IQP(A, B, Xinex, lambda_row, lambda_col, mu)
    
        print(f'| {"":<15} | {"":>8} | {"IPG-FixStep":<12} | {f_inex_val:>10.4f} | {kinex:>7} | {iinex:>7} | {"-":>7} | {time_inex:>8.2f} |')
        all_results_table.append({'Problem': name, 'Dims': f'{n}x{m}', 'Lips': Lips, 'Method': 'IPG-FixStep', 'F(X^k)': f_inex_val, 'Out-IT': kinex, 'In-IT': iinex, 'ELS-IT': '-', 'Time(s)': time_inex})
      
        c0, c1, tau_nipg, theta_1, theta_2 = 0.5, 0.49, 0.2, 1.1, 1
        start_INPGl_opt1 = time.time()
        [X_inpgl_opt1, f_inpgl_opt1, k_inpgl_opt1, i_inpgl_opt1, F_func_inpgl_opt1, time_inpgl_opt1, X_func_inpgl_opt1, step_sizes_inpgl_opt1] = NIPGl(
            A, B, X0, f_x, gradf_x, lambda_row, lambda_col,
            2000, c0=c0, c1=c1, theta_1=theta_1, theta_2=theta_2, tau=tau_nipg, 
            initial_step_size=initial_t0, option=1, FOpt=F_opt_val, mu=mu
        )
        end_INPGl_opt1 = time.time()
        time_nipg1 = end_INPGl_opt1 - start_INPGl_opt1
        f_nipg1_val = fun_F_IQP(A, B, X_inpgl_opt1, lambda_row, lambda_col, mu)
    
        print(f'| {"":<15} | {"":>8} | {"NIPG1 (A1)":<12} | {f_nipg1_val:>10.4f} | {k_inpgl_opt1:>7} | {i_inpgl_opt1:>7} | {"-":>7} | {time_nipg1:>8.2f} |')
        all_results_table.append({'Problem': name, 'Dims': f'{n}x{m}', 'Lips': Lips, 'Method': 'NIPG1 (A1)', 'F(X^k)': f_nipg1_val, 'Out-IT': k_inpgl_opt1, 'In-IT': i_inpgl_opt1, 'ELS-IT': '-', 'Time(s)': time_nipg1})

        start_INPGl_opt2 = time.time()
        [X_inpgl_opt2, f_inpgl_opt2, k_inpgl_opt2, i_inpgl_opt2, F_func_inpgl_opt2, time_inpgl_opt2, X_func_inpgl_opt2, step_sizes_inpgl_opt2] = NIPGl(
            A, B, X0, f_x, gradf_x, lambda_row, lambda_col,
            2000, c0=c0, c1=c1, theta_1=theta_1, theta_2=theta_2, tau=tau_nipg, 
            initial_step_size=initial_t0, option=2, FOpt=F_opt_val, mu=mu
        )
        end_INPGl_opt2 = time.time()
        time_nipg2 = end_INPGl_opt2 - start_INPGl_opt2
        f_nipg2_val = fun_F_IQP(A, B, X_inpgl_opt2, lambda_row, lambda_col, mu)
    
        print(f'| {"":<15} | {"":>8} | {"NIPG2 (A2)":<12} | {f_nipg2_val:>10.4f} | {k_inpgl_opt2:>7} | {i_inpgl_opt2:>7} | {"-":>7} | {time_nipg2:>8.2f} |')
        print(divider)
        all_results_table.append({'Problem': name, 'Dims': f'{n}x{m}', 'Lips': Lips, 'Method': 'NIPG2 (A2)', 'F(X^k)': f_nipg2_val, 'Out-IT': k_inpgl_opt2, 'In-IT': i_inpgl_opt2, 'ELS-IT': '-', 'Time(s)': time_nipg2})
    
        results_to_plot = [
            (F_func2, X_func_c2, time_IT_LS2, None, 'IPG-ELS', 'blue'),
            (F_funcFB, X_func_FB, time_IT_FB, None, 'PG-ELS', 'yellow'),
            (F_funcinex, X_func_FBinex, time_IT_FBinex, None, 'IPG-FixStep', 'black'),
            (F_func_inpgl_opt2, X_func_inpgl_opt2, time_inpgl_opt2, step_sizes_inpgl_opt2, 'NIPG2 (A2)', 'red'), 
            (F_func_inpgl_opt1, X_func_inpgl_opt1, time_inpgl_opt1, step_sizes_inpgl_opt1, 'NIPG1 (A1)', 'purple'),
        ]
 
        print(f'[*] Generating and Saving Plots for {name}...')
        title_details = f' (n={n}, m={m}, Lips={Lips:.2f})'   

        plot_F_vs_Time(results_to_plot, F_opt=F_opt_val, problem_name=name, title_details=title_details)
        plot_StepSizes_vs_Iterations(results_to_plot, problem_name=name, title_details=title_details)
        print(divider)
    
        keys = all_results_table[0].keys()
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results_table)

        print(f"[*] Success! All tabular results for {len(active_dimensions)} dimensions have been securely saved to: {csv_file_path}")
        
        del A, B, X0, f_x, gradf_x, F_opt_val, X_opt_val
        del X2, F_func2, time_IT_LS2, X_func_c2
        del XFB, F_funcFB, time_IT_FB, X_func_FB
        del Xinex, F_funcinex, time_IT_FBinex, X_func_FBinex
        del X_inpgl_opt1, F_func_inpgl_opt1, time_inpgl_opt1, X_func_inpgl_opt1, step_sizes_inpgl_opt1
        del X_inpgl_opt2, F_func_inpgl_opt2, time_inpgl_opt2, X_func_inpgl_opt2, step_sizes_inpgl_opt2
        del results_to_plot
        gc.collect()
    
        print(f"[*] Successfully cleared memory for dimension: {name}\n")