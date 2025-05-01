import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import tracemalloc

def fokker_planck_classical(bits, D, t_max, Nt):
    
    tracemalloc.start() #traces the memory usage
    
    # Parameters
    sigma = 0.8
    domain_factor = 10
    x_min, x_max = -domain_factor*sigma, domain_factor*sigma  # Spatial domain
    Nx = 2**bits  # Number of spatial grid points
    Dx = (x_max - x_min) / (Nx-1)  # Spatial step
    tau = t_max / Nt  # Time step
    
    #Discretized space and timea
    t = np.linspace(0, t_max, Nt + 1)  # Include final time
    x_grid = np.linspace(x_min, x_max, Nx)
    
    #Gaussian distribution
    P = scipy.stats.norm.pdf(x_grid, scale=sigma)
    #Finite difference solution
    P_all = [P.copy()]
    starttime = time.time()
    for _ in range(Nt):
        P_new = P.copy()
        P_new[1:-1] = P[1:-1] + D * tau / Dx**2 * (P[2:] - 2*P[1:-1] + P[:-2])
        P = P_new
        P_all.append(P.copy())
    endtime = time.time()
    print(f'Time for algorithm: {endtime-starttime} seconds')
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f} MB")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")
    tracemalloc.stop()
    
    return {
        'x_grid': x_grid,
        'classical_states': P_all,
        'times': t
    }

def plot_results(results, t_cut):
    x_grid = results['x_grid']
    states = results['classical_states']
    times = results['times']
    t_list = np.arange(0, t_cut, 0.4)
    t_fixed = []
    
    for t_target in t_list:
        idx = np.argmin(np.abs(np.array(times) - t_target))
        t_fixed.append(idx)
    mtlist=[]
    for i, t in enumerate(t_fixed):
        plt.plot(x_grid, states[t], '-', 
                 label=f't = {t_list[i]:.2f}')
        idx_x0 = np.argmin(np.abs(x_grid))
        mtlist.append(states[t][idx_x0])
        
    print("p(0):", mtlist)
    
    # Compute ratios of decrease
    if len(mtlist) > 1:
        ratios = [mtlist[i] / mtlist[i-1] for i in range(1, len(mtlist))]
        print("Decrease ratios:", ratios)
        
    # plt.title("Classical Finite Difference")
    plt.xlabel('x')
    plt.ylabel('P(x,t)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fokker_planck_classical(10qubits).png', dpi=300)
    plt.show()
    
results = fokker_planck_classical(
    bits=6, # just match with qubits from quantum runs
    D=0.5, #diffusion coefficient
    t_max=2, #time cutoff
    Nt=200 #time steps
)

time_cutoff = 2
plot_results(results, time_cutoff)
