"""CUR-Like Factorization optimization problem"""

import numpy as np
from math import sqrt
from fun_gradobjec import fun_grad_f_CUR, fun_F_CUR
from algos_cur import IPG_ELS, PG_ELS, IPG_Fixstep, NIPGl
import time
import matplotlib.pyplot as plt
import os
import csv
import gc

# --- SETUP DIRECTORIES FOR CUR ---
BASE_DIR = 'result/CUR'
IMG_DIR = os.path.join(BASE_DIR, 'images')
CSV_DIR = os.path.join(BASE_DIR, 'csv_results')
DATA_DIR = os.path.join(BASE_DIR, 'saved_data')

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def save_experiment_data(dataset_name, mu, rho, output_dir=DATA_DIR, **data_dict):
    """
    Save all experimental results into a compressed .npz file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"results_{dataset_name}_rho{rho}_mu{mu}.npz"
    filepath = os.path.join(output_dir, filename)
    
    np.savez(filepath, **data_dict)
    print(f"[*] Successfully packaged and saved data at: {filepath}")

def get_safe_filename(title_str):
    """Helper function to convert a plot title into a valid cross-platform filename."""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '$']
    safe_title = title_str
    for char in invalid_chars:
        safe_title = safe_title.replace(char, '')
    return safe_title.replace(' ', '_') + '.png'

def plot_F_vs_Time(algorithm_results, F_opt=None, problem_name="", title_details=""):
    """Plot F(Xk) - F* convergence over TIME"""
    plt.figure(figsize=(10, 8)) 
    
    markers = ['o', '*', '^', 's', 'D', 'v'] 
    
    for idx, (F_func, X_func, time_func, step_sizes, label, color) in enumerate(algorithm_results):
        F_to_plot = np.array(F_func)
        T_axis = np.array(time_func)
        
        current_marker = markers[idx % len(markers)]
        line_kwargs = {'label': label, 'color': color, 'marker': current_marker, 'markevery': 0.1, 'markersize': 10, 'linewidth': 3}
        
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
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    
def plot_StepSizes_vs_Iterations(algorithm_results, problem_name="", title_details=""):
    """Plot Adaptive Step Size t_k over ITERATIONS"""
    plt.figure(figsize=(10, 8))
    
    markers = ['o', '*', '^', 's', 'D', 'v'] 
    
    for idx, (F_func, X_func, time_func, step_sizes, label, color) in enumerate(algorithm_results):
        if step_sizes is not None and len(step_sizes) > 1:
            iterations = np.arange(len(step_sizes))
            
            current_marker = markers[idx % len(markers)]
            plt.semilogy(iterations, step_sizes, label=label, color=color, marker=current_marker, markevery=0.1, markersize=10, linewidth=3)
        
    title_str = f'{problem_name} - Adaptive Step Size $t_k$ over Iterations' + title_details
    plt.title(title_str)
    plt.xlabel('Outer Iterations (k)')
    plt.ylabel(r'Step Size $t_k$ (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    filepath = os.path.join(IMG_DIR, get_safe_filename(title_str))
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

prec: float = 1.0e-3
kmax = 100
lambda_row = lambda_col = 0.01
mu =0 # 0.01
rho = 4
problem_type = "Group Elastic Net" if mu > 0 else "Group Lasso"

csv_filename = os.path.join(CSV_DIR, f"results_{problem_type}_mu{mu}_rho{rho}_alpha=0.5)_cina0.csv")
with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Problem', 'Lips', 'Method', 'F(X^k)', 'Out-IT', 'In-IT', 'ELS-IT', 'Time (s)'])
print(f"\n[*] Running {problem_type} | mu = {mu} | rho = {rho}")
print("-" *  109)
print(f"| {'Problem':<20} | {'Lips':>10} | {'Method':<13} | {'F(X^k)':>10} | {'Out-IT':>7} | {'In-IT':>7} | {'ELS-IT':>7} | {'Time (s)':>10} |")
print("-" *  109)

for i in range(1, 7, 1):
    if i == 1:
        W = np.loadtxt('CSV/ColonTumor_62x2000.txt', delimiter=',')
        name = 'Colon Tumor'
    if i == 2:
        W = np.loadtxt('CSV/heart_303x14.txt', delimiter=',')
        name = 'Heart Disease'
    if i == 3:
        W = np.loadtxt('CSV/CentralNervousSystem_60x7129.txt', delimiter=',')
        name = 'CNS'
    if i == 4:
        W = np.loadtxt('CSV/LungCancer-Michigan_96x7129.txt', delimiter=',')
        name = 'Lung cancer-Michigan'
    if i == 5:
        W = np.loadtxt('CSV/Secom_A_1567x590.txt', delimiter=',')
        name = 'Secom'
    if i == 6:
        W = np.loadtxt('CSV/cina0_16033x132_d.txt', delimiter=',')
        name = 'Cina0'
        
    W = W - np.mean(W)
    W = rho * W / (np.linalg.norm(W, 'fro'))
    
    Lips = np.linalg.norm(np.dot(np.transpose(W), W)) ** 2.
    initial_t0 = 0.0001
    X0 = np.zeros((len(W[0]), len(W)))
    
    f_x, gradf_x = fun_grad_f_CUR(W, X0)
    
    gamma_1 = 1.1
    gamma_2 = 1.1
    theta = 0.5
    tau = 0.8
    alpha = 0.01
    t_k = 1
    startFBinexLS2 = time.time()
    [X2, f_X2, k2, i2, ItLiSear2, F_func2, time_IT_LS2, X_func_c2] = IPG_ELS(W, X0, f_x, gradf_x, gamma_1, gamma_2, theta, tau, alpha, t_k, kmax, prec, lambda_row, lambda_col, mu=mu)
    endFBinexLS2 = time.time()
    core_time_FBinex2 = time_IT_LS2[-1]
    print(f"| {name:<20} | {Lips:>10.4f} | {'IPG-ELS':<13} | {fun_F_CUR(W, X2, lambda_row, lambda_col, mu):>10.4f} | {k2:>7} | {i2:>7} | {ItLiSear2:>7} | {core_time_FBinex2:>10.2f} |")
    
    theta = 0.5
    t_k = 1
    startFBLS = time.time()
    [XFB, f_XFB, kFB, iFB, ItLiSearFB, F_funcFB, time_IT_FB, X_func_FB] = PG_ELS(W, X0, f_x, gradf_x, theta, t_k, 2000, prec, lambda_row, lambda_col, fun_F_CUR(W, X2, lambda_row, lambda_col, mu), mu=mu)
    endFBLS = time.time()
    core_time_FBLS = time_IT_FB[-1]
    print(f"| {' ':<20} | {' ':>10} | {'PG-ELS':<13} | {fun_F_CUR(W, XFB, lambda_row, lambda_col, mu):>10.4f} | {kFB:>7} | {iFB:>7} | {ItLiSearFB:>7} | {core_time_FBLS:>10.2f} |")
    
    sigma = 0.9
    t_k = 1/Lips
    gamma = 1
    startFBinex = time.time()
    [Xinex, f_Xinex, kinex, iinex, F_funcinex, time_IT_FBinex, X_func_FBinex] = IPG_Fixstep(W, X0, f_x, gradf_x, sigma, t_k, gamma, 2000, prec, lambda_row, lambda_col, fun_F_CUR(W, X2, lambda_row, lambda_col, mu), mu=mu)
    endFBinex = time.time()
    core_time_FBinex = time_IT_FBinex[-1]
    print(f"| {' ':<20} | {' ':>10} | {'IPG-FixStep':<13} | {fun_F_CUR(W, Xinex, lambda_row, lambda_col, mu):>10.4f} | {kinex:>7} | {iinex:>7} | {'-':>7} | {core_time_FBinex:>10.2f} |")
    
    c0 = 0.5 
    c1 = 0.49
    tau = 0.2
    theta_1 = 1.1
    theta_2 = 1

    start_INPGl_opt1 = time.time()
    [X_inpgl_opt1, f_inpgl_opt1, k_inpgl_opt1, i_inpgl_opt1, F_func_inpgl_opt1, time_inpgl_opt1, X_func_inpgl_opt1, step_sizes_inpgl_opt1] = NIPGl(
        W, X0, f_x, gradf_x, lambda_row, lambda_col,
        kmax=2000,
        c0=c0, c1=c1,
        theta_1=theta_1, theta_2=theta_2, tau=tau, 
        initial_step_size=initial_t0,
        option=1, 
        FOpt=fun_F_CUR(W, X2, lambda_row, lambda_col, mu),
        mu=mu
    )
    end_INPGl_opt1 = time.time()
    core_time_inpgl_opt1 = time_inpgl_opt1[-1]

    print(f"| {' ':<20} | {' ':>10} | {'NIPG1 (A1)':<13} | {fun_F_CUR(W, X_inpgl_opt1, lambda_row, lambda_col, mu):>10.4f} | {k_inpgl_opt1:>7} | {i_inpgl_opt1:>7} | {'-':>7} | {core_time_inpgl_opt1:>10.2f} |")
    
    c0 = 0.5
    c1 = 0.49
    tau = 0.2
    theta_1 = 1.1
    theta_2 = 1

    start_INPGl_opt2 = time.time()
    [X_inpgl_opt2, f_inpgl_opt2, k_inpgl_opt2, i_inpgl_opt2, F_func_inpgl_opt2, time_inpgl_opt2, X_func_inpgl_opt2, step_sizes_inpgl_opt2] = NIPGl(
        W, X0, f_x, gradf_x, lambda_row, lambda_col,
        kmax=2000,
        c0=c0, c1=c1,
        theta_1=theta_1, theta_2=theta_2, tau=tau, 
        initial_step_size=initial_t0,
        option=2,
        FOpt=fun_F_CUR(W, X2, lambda_row, lambda_col, mu),
        mu=mu
    )
    end_INPGl_opt2 = time.time()
    core_time_inpgl_opt2 = time_inpgl_opt2[-1]

    print(f"| {' ':<20} | {' ':>10} | {'NIPG2 (A2)':<13} | {fun_F_CUR(W, X_inpgl_opt2, lambda_row, lambda_col, mu):>10.4f} | {k_inpgl_opt2:>7} | {i_inpgl_opt2:>7} | {'-':>7} | {core_time_inpgl_opt2:>10.2f} |")
    print("-" *  109)
    csv_rows = [
        [name, f"{Lips:.4f}", "IPG-ELS", f"{fun_F_CUR(W, X2, lambda_row, lambda_col, mu):.4f}", k2, i2, ItLiSear2, f"{core_time_FBinex2:.2f}"],
        ["", "", "PG-ELS", f"{fun_F_CUR(W, XFB, lambda_row, lambda_col, mu):.4f}", kFB, iFB, ItLiSearFB, f"{core_time_FBLS:.2f}"],
        ["", "", "IPG-FixStep", f"{fun_F_CUR(W, Xinex, lambda_row, lambda_col, mu):.4f}", kinex, iinex, "-", f"{core_time_FBinex:.2f}"],
        ["", "", "NIPG1 (A1)", f"{fun_F_CUR(W, X_inpgl_opt1, lambda_row, lambda_col, mu):.4f}", k_inpgl_opt1, i_inpgl_opt1, "-", f"{core_time_inpgl_opt1:.2f}"],
        ["", "", "NIPG2 (A2)", f"{fun_F_CUR(W, X_inpgl_opt2, lambda_row, lambda_col, mu):.4f}", k_inpgl_opt2, i_inpgl_opt2, "-", f"{core_time_inpgl_opt2:.2f}"]
    ]
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    results_to_plot = [
       
        (F_func2, X_func_c2, time_IT_LS2, None, 'IPG-ELS', 'blue'),
        (F_funcFB, X_func_FB, time_IT_FB, None, 'PGLS', 'yellow'),
        (F_funcinex, X_func_FBinex, time_IT_FBinex, None, 'IPGFixstep', 'black'),
        (F_func_inpgl_opt2, X_func_inpgl_opt2, time_inpgl_opt2, step_sizes_inpgl_opt2, 'NIPG2', 'red'), 
        (F_func_inpgl_opt1, X_func_inpgl_opt1, time_inpgl_opt1, step_sizes_inpgl_opt1, 'NIPG1', 'purple'),
    ]
 
    print('\nGenerating Plots...')
    problem_name = name 
    title_details = rf' ($\mu$={mu}, $\rho$={rho}, Lips={Lips:.2f})'
    
    F_opt_val = fun_F_CUR(W, X2, lambda_row, lambda_col, mu)
    
    plot_F_vs_Time(results_to_plot, F_opt=F_opt_val, problem_name=problem_name, title_details=title_details)
    plot_StepSizes_vs_Iterations(results_to_plot, problem_name=problem_name, title_details=title_details)

    save_experiment_data(
        dataset_name=name,   
        rho=rho,    
        mu=mu,         
        output_dir=DATA_DIR,
        
        plot_title=name, 
        mu_value=mu,
        lipschitz_constant=Lips,
        
        time_IPG_ELS=time_IT_LS2,
        F_IPG_ELS=F_func2,
           
        time_PG_ELS=time_IT_FB,
        F_PG_ELS=F_funcFB,
             
        time_IPG_FixStep=time_IT_FBinex,
        F_IPG_FixStep=F_funcinex,
     
        time_NIPG1=time_inpgl_opt1,
        F_NIPG1=F_func_inpgl_opt1,
        step_NIPG1=step_sizes_inpgl_opt1, 
        
        time_NIPG2=time_inpgl_opt2,
        F_NIPG2=F_func_inpgl_opt2,
        step_NIPG2=step_sizes_inpgl_opt2  
    )
    
    del W, X0, f_x, gradf_x
    del X2, F_func2, time_IT_LS2, X_func_c2
    del XFB, F_funcFB, time_IT_FB, X_func_FB
    del Xinex, F_funcinex, time_IT_FBinex, X_func_FBinex
    del X_inpgl_opt1, F_func_inpgl_opt1, time_inpgl_opt1, X_func_inpgl_opt1, step_sizes_inpgl_opt1
    del X_inpgl_opt2, F_func_inpgl_opt2, time_inpgl_opt2, X_func_inpgl_opt2, step_sizes_inpgl_opt2
    del results_to_plot
    
    gc.collect()
    
    print(f"[*] Successfully cleared memory for dataset: {name}\n")