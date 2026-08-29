import os
import numpy as np
import matplotlib.pyplot as plt

def setup_plot_style():
    """Configure general styling for all plots to match the academic format."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'lines.linewidth': 3,
        'lines.markersize': 5
    })

def get_safe_filename(title_str):
    """Helper function to convert a plot title into a valid cross-platform filename."""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '$']
    safe_title = title_str
    for char in invalid_chars:
        safe_title = safe_title.replace(char, '')
    return safe_title.replace(' ', '_') + '.png'

def plot_step_size(iterations_nipg1, step_nipg1, iterations_nipg2, step_nipg2, dataset_name, mu, rho, lips):
    """Recreates: Adaptive Step Size t_k over Iterations"""
    plt.figure(figsize=(10, 8))
    
    # NIPG2 is red (square 's'), NIPG1 is purple (diamond 'D').
    plt.semilogy(iterations_nipg2, step_nipg2, label='NIPG2', color='red', marker='s', markevery=0.05, markersize=8)
    plt.semilogy(iterations_nipg1, step_nipg1, label='NIPG1', color='purple', marker='D', markevery=0.05, markersize=8)
    
    plt.xlabel('Outer Iterations (k)')
    plt.ylabel(r'Step Size $t_k$ (Log Scale)')
    title_str = rf'{dataset_name} - Adaptive Step Size $t_k$ over Iterations ($\mu$={mu}, rho={rho}, Lips={lips:.2f})'
    plt.title(title_str)
    
    # Grid setup: major and minor dashed lines
    plt.grid(True, which='major', linestyle='--', color='gray', alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', color='gray', alpha=0.4)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_convergence_F(time_dict, F_dict, F_star, dataset_name, mu, rho, lips):
    """Recreates: Convergence of F over Time"""
    plt.figure(figsize=(10, 8))
    
    # Specific styles matching algorithms with distinct markers
    styles = {
        'IPG-ELS': {'color': 'blue', 'marker': 'o'},      # Circle
        'PGLS': {'color': 'yellow', 'marker': '*'},       # Star
        'IPGFixstep': {'color': 'black', 'marker': '^'},  # Triangle Up
        'NIPG2': {'color': 'red', 'marker': 's'},         # Square
        'NIPG1': {'color': 'purple', 'marker': 'D'},      # Diamond
        'Tseng': {'color': 'brown', 'marker': 'v'}        # Triangle Down
    }
    
    for algo, F_vals in F_dict.items():
        if algo in time_dict and algo in styles:
            time_vals = time_dict[algo]
            # Calculate F(X_k) - F*. Use np.maximum to avoid log(0)
            F_diff = np.maximum(np.array(F_vals) - F_star, 1e-16)
            
            plt.semilogy(time_vals, F_diff, label=algo, 
                         color=styles[algo]['color'], 
                         marker=styles[algo]['marker'], markevery=0.1, markersize=8)

    plt.xlabel('Time (seconds)')
    plt.ylabel(r'$F(X_k) - F^*$ (Log Scale)')
    title_str = rf'{dataset_name} - Convergence of F over Time ($\mu$={mu}, rho={rho}, Lips={lips:.2f})'
    plt.title(title_str)
    
    plt.grid(True, which='major', linestyle='--', color='gray', alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', color='gray', alpha=0.4)
    plt.legend(loc='center right')
    
    plt.tight_layout()
    plt.show()



# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    setup_plot_style()
        
    filepath = 'result/CUR\saved_data\results_Colon Tumor_rho4_mu0.01.npz'
    
    if os.path.exists(filepath):
        data = np.load(filepath)
        
        # 1. Extract metadata
        dataset_name = str(data['plot_title'].item())
        dataset_name_display = dataset_name.replace('_', ' ') 
        
        # --- DECLARE PARAMETERS HERE ---
        rho = 4.0 
        mu = 0.01 # You can change this to 0 if plotting for Group Lasso
        lips = float(data['lipschitz_constant'].item())
        
        print(f"[*] Loaded data for: {dataset_name_display} | mu={mu} | rho={rho} | Lips={lips:.4f}")
        
        # 2. Extract step sizes for NIPG1 & NIPG2
        step_nipg1 = data['step_NIPG1']
        iterations_nipg1 = np.arange(len(step_nipg1))
        
        step_nipg2 = data['step_NIPG2']
        iterations_nipg2 = np.arange(len(step_nipg2))
        
        # 3. Map saved variables to legend names
        algo_mapping = {
            'IPG-ELS': ('time_IPG_ELS', 'F_IPG_ELS', 'X_IPG_ELS'),
            'PGLS': ('time_PG_ELS', 'F_PG_ELS', 'X_PG_ELS'),
            'IPGFixstep': ('time_IPG_FixStep', 'F_IPG_FixStep', 'X_IPG_FixStep'),
            'Tseng': ('time_Tseng', 'F_Tseng', 'X_Tseng'),
            'NIPG1': ('time_NIPG1', 'F_NIPG1', 'X_NIPG1'),
            'NIPG2': ('time_NIPG2', 'F_NIPG2', 'X_NIPG2')
        }
        
        # Ensure the order matches the legend
        plot_order = ['IPG-ELS', 'PGLS', 'IPGFixstep', 'NIPG2', 'NIPG1', 'Tseng']
        
        time_dict = {}
        F_dict = {}
        dist_dict = {}
        
        print("[*] Setting IPG-ELS as the common baseline...")
        # Establish the baseline from IPG-ELS
        if 'X_IPG_ELS' in data and 'F_IPG_ELS' in data:
            X_final_common = data['X_IPG_ELS'][-1]
            F_star_common = np.min(data['F_IPG_ELS'])
        else:
            print("[!] Warning: IPG-ELS data not found. Check your saved data.")
            X_final_common = None
            F_star_common = 0.0

        print("[*] Processing arrays and calculating distances...")
        for legend_name in plot_order:
            if legend_name in algo_mapping:
                time_key, F_key, X_key = algo_mapping[legend_name]
                if time_key in data and F_key in data and X_key in data:
                    time_dict[legend_name] = data[time_key]
                    F_dict[legend_name] = data[F_key]
                    
                    X_array = data[X_key]
                    
                    # CALCULATE DISTANCES BASED ON THE COMMON IPG-ELS SOLUTION
                    if X_final_common is not None:
                        distances = [np.linalg.norm(X - X_final_common, 'fro') for X in X_array]
                        dist_dict[legend_name] = np.array(distances)
        
        # SET F* BASED ON THE COMMON IPG-ELS SOLUTION
        F_star = F_star_common
        
        # 4. Generate and display the plots
        print("[*] Displaying Adaptive Step Size Plot...")
        plot_step_size(iterations_nipg1, step_nipg1, iterations_nipg2, step_nipg2, dataset_name_display, mu, rho, lips)
        
        print("[*] Displaying Convergence of F Plot...")
        plot_convergence_F(time_dict, F_dict, F_star, dataset_name_display, mu, rho, lips)
        
        
        
        print("[*] All plots displayed successfully for analysis!")
    else:
        print(f"[!] ERROR: Data file not found at: {filepath}")