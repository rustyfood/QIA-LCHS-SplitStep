import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import scipy 
import time
import tracemalloc

def fokker_planck_lchs(n_qubits, mu, D, desiredtimes):
    """
    LCHS algorithm for Fokker-Planck 
    """
    tracemalloc.start() #traces the memory usage
    # Initial condition: Gaussian distribution
    sigma =0.8 #scale is the width of gaussian
    def initial_condition(x):
        return scipy.stats.norm.pdf(x, scale=sigma) # Scale = width
    
    # Parameters
    domain_factor = 10
    n_points = 2**n_qubits
    x_min, x_max = -domain_factor*sigma,domain_factor*sigma
    dx = (x_max - x_min) / (n_points-1)
    x_grid = np.linspace(x_min, x_max, n_points)
    
    n_control = 2 # Ancilla qubits
    total_qubits = n_qubits + n_control
    control_wires = range(n_control) # Control qubits
    target_wires = range(n_control, total_qubits) # Target qubits
    
    dev = qml.device("default.qubit", wires=total_qubits)
    
    # Shift matrices (non-quantum)
    S_plus = np.diag(np.ones(n_points-1, dtype=complex), k=1) # S+ shift
    S_minus = np.diag(np.ones(n_points-1, dtype=complex), k=-1) # S- shift
    I = np.eye(n_points, dtype=complex)
    
    # Quantum version of our shift operators + ident
    def plus_op():
        return qml.QubitUnitary(S_plus, wires=target_wires)
        
    def minus_op():
        return qml.QubitUnitary(S_minus, wires=target_wires)
        
    def I_op():
        return qml.QubitUnitary(-I, wires=target_wires)
    
    ops = [plus_op(), I_op(), minus_op()]
                
    # Coefficients - need to be positive for LCHS
    coeffs = np.array([
        D/(dx**2),      # S+ term
        2*D/(dx**2),    # I term
        D/(dx**2)       # S- term
    ])
    
    hamiltonian = qml.Hamiltonian(coeffs, ops)
    
    @qml.qnode(dev)
    def fokker_planck_step(state_vector,timez):
        # Prep initial state on target qubits
        qml.StatePrep(state_vector, wires=target_wires, normalize=True)
        
        qml.evolve(hamiltonian, timez)
        # Apply LCHS
        qml.PrepSelPrep(hamiltonian, control_wires)
                
        return qml.probs(wires=target_wires)
    
    # Evolve using LCHS
    psi_0 = initial_condition(x_grid) # Initial function
    q_states = [psi_0]
    starttime = time.time()

    for step in desiredtimes:
        # Quantum evolution        
        # Extract target state when control is |0⟩
        current_state = fokker_planck_step(psi_0.copy(),step)
        current_state = np.sqrt(current_state)
        q_states.append(current_state)
        
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

# Plot the results
def plot_results(results):
    x_grid = results['x_grid']
    quantum_states = results['quantum_states']
    times = results['times']
    
    mtlist = []
    for i, t in enumerate(times):
        plt.plot(x_grid, quantum_states[i], '-', label=f't = {times[i]:.2f}')
        # Append probability at x = 0 to mtlist
        idx_x0 = np.argmin(np.abs(x_grid))
        mtlist.append(quantum_states[i][idx_x0])
    
    print("p(0) for each time:", mtlist)
    
    # Compute ratios of decrease
    if len(mtlist) > 1:
        ratios = [mtlist[i] / mtlist[i-1] for i in range(1, len(mtlist))]
        print("Decrease ratios:", ratios)
    
    plt.xlabel('x')
    plt.ylabel('P(x,t)')
    plt.legend()
    plt.grid(True)
    # plt.title('LCHS')
    plt.savefig('fokker_planck_lchs(10qubits).png', dpi=300)
    plt.show()

results = fokker_planck_lchs(
    n_qubits=6,         # Target qubits
    mu=0,               # No drift term
    D=0.5,              # Diffusion coefficient
    desiredtimes=[0.4,0.8,1.2,1.6]   # manually entering which times to view
)

# Plot the results
plot_results(results)