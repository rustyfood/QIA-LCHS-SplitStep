import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import scipy
import time
import tracemalloc

def fokker_planck_spectral_split_step(n_qubits, desiredtimes, mu, D):
    """
    Fokker-Planck solution using the spectral split-step method.
    
    Args:
        n_qubits: Total number of qubits
        desiredtimes: Time evolutions we want to simulate
        mu: Drift coefficient (can be a constant or function of x)
        D: Diffusion coefficient (can be a constant or function of x)
    """
    tracemalloc.start() #traces the memory usage
    dev = qml.device("default.qubit", wires=n_qubits)
    
    sigma = 0.8
    domain_factor = 10
    n_points = 2**n_qubits #amount of nodes
    x_min, x_max = -domain_factor*sigma,domain_factor*sigma
    dx = (x_max - x_min) / (n_points-1)
    x_grid = np.linspace(x_min, x_max, n_points)
    
    #Calculate momentum grid values
    k_grid = 2*np.pi*np.fft.fftfreq(n_points, dx)    
    #Gaussian distribution
    def initial_condition(x):
        return scipy.stats.norm.pdf(x, scale=sigma) #scale = width
    
    #position space: e^(-i*H_drift*dt/2) = 0 (for Wiener Process)
        #drift would also be dealt with in momentum space
    
    #momentum space: e^(-i*H_diffusion*dt)
    def diffusion_phases(k):
        return -D * (k**2)
    
    @qml.qnode(dev)
    def fokker_planck_step(state_vector,timez):
        qml.StatePrep(state_vector, wires=range(n_qubits), normalize=True)
        #QFT to momentum space
        qml.QFT(wires=range(n_qubits))
        # Apply diffusion in momentum space
        generator = diffusion_phases(k_grid) #only -D *k^2 for Wiener process
        diagonal_elements = np.exp(1j * generator*timez) #the exponential of FT , dt is time slice
        qml.DiagonalQubitUnitary(diagonal_elements, wires=range(n_qubits)) #allows operator to be applied
        
        #Inverse QFT to position space
        qml.adjoint(qml.QFT(wires=range(n_qubits)))
            
        return qml.probs(wires=range(n_qubits))

    # Time evolution
    psi_0 = initial_condition(x_grid) #making the gaussian depn on qubits
    q_states = [psi_0.copy()]
    starttime = time.time()

    for step in desiredtimes:
        # Quantum evolution
        current_state = fokker_planck_step(psi_0.copy(),step)
        q_states.append(np.sqrt(current_state))
    endtime = time.time()
    
    print(f'Time for algorithm: {endtime-starttime} seconds')
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f} MB")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")
    tracemalloc.stop()
    return {
        'x_grid': x_grid,
        'quantum_states': q_states,
        'times': np.hstack([0,desiredtimes])
        }

#plotting function
def plot_fokker_planck_results(results):
    '''Args:
    results: the results from running the quantum algorithm
    t_cut: the time cutoff for observations on the graph
    '''
    x_grid = results['x_grid']
    quantum_states = results['quantum_states']
    times = results['times']
        
    mtlist=[]
    for i, t in enumerate(times):
        plt.plot(x_grid, quantum_states[i], '-', 
                 label=f't = {times[i]:.2f}')
        # Append probability at x = 0 to mtlist
        idx_x0 = np.argmin(np.abs(x_grid))
        mtlist.append(quantum_states[i][idx_x0])
        
    print("p(0) for each time:", mtlist)
    
    # Compute ratios of decrease
    if len(mtlist) > 1:
        ratios = [mtlist[i] / mtlist[i-1] for i in range(1, len(mtlist))]
        print("Decrease ratios:", ratios)
        
    # plt.title("SpectralSplit")
    plt.xlabel('x')
    plt.ylabel('P(x,t)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fokker_planck_spectral(10qubits).png', dpi=300)
    plt.show()

results = fokker_planck_spectral_split_step(
    n_qubits=6, #total qubits 
    desiredtimes=[0.4,0.8,1.2,1.6], #specific times we want to observe
    mu=0.0,  # drift (pure diffusion case)
    D=0.5    #Diffusion coefficient
)

plot_fokker_planck_results(results)
