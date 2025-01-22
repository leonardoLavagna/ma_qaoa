#------------------------------------------------------------------------------
# OPTIMIZATION ROUTINES FOR MA-QAOA INSTANCES
# @ Leonardo Lavagna 2024
#------------------------------------------------------------------------------


import random 
import numpy as np
from typing import List, Tuple
from networkx import Graph
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from functions import maxcut_utilities as m_utils
from config import backend, verbose


num_evaluations = 0


def objective_function(init_point, circuit):
    # Setup
    G = circuit.G
    qc = circuit.get_circuit()
    #print(qc.num_parameters)
    qc = qc.assign_parameters(init_point)
    # Executing the circuit to get the energies...
    t_qc = transpile(qc, backend=backend)
    job = backend.run(t_qc)
    counts = job.result().get_counts(qc)
    # Getting the results...
    energy = m_utils.compute_maxcut_energy(G, m_utils.invert_counts(counts), verbose=verbose)
    
    return -energy
        

def callback(x: List):
    """
    Function to be called at each iteration of the optimization step (in the minimize method) to count the number of optimization steps.

    Args:
        x (list): current solution of the optimization step
    """
    global num_evaluations
    num_evaluations += 1


def simple_optimization(circuit, method: str = 'COBYLA', seed: int = None, verbose: bool = True) -> Tuple[np.ndarray, float]:
    """
    Perform a simple optimization routine.

    Args:
        circuit (QuantumCircuit): Quantum circuit.
        method (str): Type of optimizer. The default is 'COBYLA'.
        seed (int): Seed needed for reproducibility. The default is None.
        verbose (bool): If True enters in debugging mode. The default is False.

    Returns:
        tuple: Optimized parameters and corresponding objective function value.
    """
    # Setup
    betas = circuit.betas
    gammas = circuit.gammas
    init_point = list(betas) + list(gammas)
    if verbose:
        print(" --------------------------------- ")
        print("| Parameters for the optimization. |".upper())
        print(" --------------------------------- ")
        print("\t * betas:", betas)
        print("\t * gammas:", gammas)
        print("\t * init_point:", init_point)
    
    # Optimizing... 
    if verbose is True:
        print(" --------------- ")
        print("| Optimizing... |".upper())
        print(" --------------- ")
        optimizer = minimize(objective_function, init_point, args=(circuit), callback=callback, method=method)   
    elif verbose is False:
        optimizer = minimize(objective_function, init_point, args=(circuit), method=method)
        
    # Getting the results...  
    optimal_value = -optimizer.fun
    optimal_angles = optimizer.x
    if verbose:
        print(" --------- ")
        print("| Results. |".upper())
        print(" --------- ")
        print("\t * optimal_anlges:", optimal_angles)
        print("\t * optimal_value:", optimal_value)
    
    return optimizer.x, optimizer.fun