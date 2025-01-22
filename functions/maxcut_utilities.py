#------------------------------------------------------------------------------
# MAXCUT UTILITIES
# @ Leonardo Lavagna 2024
#------------------------------------------------------------------------------


import random 
import numpy as np
from typing import List, Tuple
from networkx import Graph
from itertools import chain
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from config import verbose


def compute_max_cut_exactly(G: Graph) -> int:
    """Calculates the max-cut of the graph G with a brute-force approach.

    Args:
        G (Graph): The graph for which the max-cut needs to be calculated. Note G should be a graph created 
                   with the Problems class

    Returns:
        int: The maximum cut value obtained through brute-force.
    """
    best_cost_brute = 0
    n = G.get_number_of_nodes()
    w = G.get_adjacency_matrix()
    for b in range(2**n):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
        cost = 0
        for i in range(n):
            for j in range(n):
                cost = cost + w[i,j]*x[i]*(1-x[j])
        if best_cost_brute < cost:
            best_cost_brute = cost
            xbest_brute = x
            
    return int(best_cost_brute)


def invert_counts(counts: dict) -> dict:
    """
    Invert the keys of a dictionary.

    Args:
        counts (dict): Input dictionary with keys and values.

    Returns:
        dict: A new dictionary with inverted keys.
    """
    rearranged_counts = {k[::-1]: v for k, v in counts.items()}
    
    return rearranged_counts


def get_maxcut_number_of_edges(G: Graph, x: List) -> int:
    """
    Calculate the number of edges in the maximum cut of a graph for a given node assignment.

    Args:
        G (Graph): Input graph.
        x (list): Node assignment.

    Returns:
        int: Number of edges in the maximum cut.
        
    Note: 
        The cut value starts from zero and is decreased at each step, resulting in a final negative cut value.
    """
    cut = 0
    for i, j in G.get_edges():
        if x[i] != x[j]:
            cut -= 1
    
    return cut


def compute_maxcut_energy(G: Graph, counts: dict, verbose: bool = False) -> float:
    """
    Compute the energy of the maximum cut for a given graph and measurement counts.

    Args:
        G (Graph): Input graph.
        counts (dict): Measurement counts.
        verbose (bool): If True enters in debugging mode. The default is False.

    Returns:
        float: Energy of the maximum cut.
    """
    # Setup
    energy = 0
    total_counts = 0

    # Getting the energies...
    for meas, meas_count in counts.items():
        obj_for_meas = get_maxcut_number_of_edges(G, meas)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    if verbose is True:
        max_cut_state = max(counts, key=lambda k: counts[k])        
        print(f"\t\t Current MaxCut state and value: state = {max_cut_state},  value = {-(energy / total_counts)}.")
    approximate_energy = energy / total_counts
    
    return -approximate_energy