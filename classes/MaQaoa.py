#------------------------------------------------------------------------------
# CLASS MULTI-ANGLE QAOA FOR THE MAXCUT PROBLEM
# @ Leonardo Lavagna 2024
#------------------------------------------------------------------------------


import numpy as np
from matplotlib import pyplot as plt
from classes import Problems as P
from functions import ma_qaoa_utilities as utils
from qiskit_aer import AerSimulator
from typing import List, Tuple
from networkx import Graph
from qiskit.circuit import ParameterVector, QuantumCircuit


class MaQaoa:
    def __init__(self, p: int=1, G: Graph = None, betas: List[float] = None, gammas: List[float] = None,
                 backend = AerSimulator(method='automatic'), measure: bool = True, 
                 seed: int = None, verbose: bool = True):
        """Initialize class MaQaoa.
        
        Args:
            p (int): Number of MaQaoa layers. The default is one.
            G (Graph): A graph created with the Problem class used as MaxCut problem instance. The default is None.
            betas (list): Angles for the mixer operator.
            gammas (list): Angles for the cost operator.
            backend (Qiskit backend): Qiskit backend to execute the code on a quantum simulator. 
            measure (bool): If True measure the qaoa circuit. The default is True.
            seed (int): Seed for a pseudo-random number generator. The default is None.
            verbose (bool): If True enters in debugging mode. The default is True.
        """
        # Setup
        self.p = p
        self.G = G
        #self.mixer = mixer
        self.backend = backend
        self.measure = measure
        self.verbose = verbose  
        self.seed = seed
        self.problems_class = P.Problems(p_type="custom", G=self.G)
        self.params = None
        if self.seed is not None:
            np.random.seed(self.seed)  
        if self.G is None:
            self.N = 0
            self.w = [[]]
            self.betas = []
            self.gammas = []
        if self.G is not None:
            self.N = G.get_number_of_nodes()
            self.M = G.get_number_of_edges()
            self.w = G.get_adjacency_matrix()
            if betas is None or gammas is None:
                self.betas = utils.generate_parameters(n=self.N*self.p, k=1) 
                self.gammas = utils.generate_parameters(n=self.M*self.p, k=2)
            if betas is not None and gammas is not None:
                self.betas = betas
                self.gammas = gammas
        
        # Checking...
        if self.problems_class.__class__.__name__ != self.G.__class__.__name__ and G is not None:
            raise Exception("Invalid parameters. The graph G should be created with the Problems class.")      
        if len(self.betas) != self.N * self.p or len(self.gammas) != self.M * self.p:
            raise ValueError("Invalid angles list. At each layer there should be a beta angle for each node and a gamma angle for each edge.")
            
        # Initializing...
        if self.verbose is True:            
            print(" --------------------------- ")
            print("| Intializing MaQaoa class... |".upper())
            print(" --------------------------- ")     
            print("-> Getting problem instance...".upper())
            if self.G is not None:
                self.G.get_draw()
                plt.show()
            if self.G is None:
                print("\t * G = ø")
            if self.betas is None and self.G is not None:
                print("-> Beta angles not provided. Generating angles...".upper())
                print(f"\t * betas = {self.betas}")
            if self.gammas is None and self.G is not None:
                print("-> Gamma angles not provided. Generating angles...".upper())
                print(f"\t * gammas = {self.gammas}")
            print("-> Getting the ansatz...".upper())
            if self.G is not None:
                print(self.get_circuit())
            if self.G is None:
                print("\t  * MaQaoa circuit = ø")
            print("-> The MaQaoa class was initialized with the following parameters.".upper())
            if self.G is None:
                print(f"\t * Graph: G = ø;")
            if self.G is not None:
                print(f"\t * Graph: G = {self.G.p_type};")
            print("\t * Angles:") 
            print(f"\t\t - betas = {self.betas};")
            print(f"\t\t - gammas = {self.gammas};")
            #print(f"\t * Mixer Hamiltonian type: '{self.mixer}';")
            print(f"\t * Random seed: seed = {self.seed};")
            print(f"\t * Measurement setting: measure = {self.measure}.")

    
    def cost_operator(self, gammas: List[float]) -> QuantumCircuit:
        """Create an instance of the cost operator with angle 'gamma'.
        
        Args:
            gammas (list): Angle for the cost operator.

        Returns:
            QuantumCircuit: Circuit representing the cost operator.
        """
        qc = QuantumCircuit(self.N, self.N)
        count = 0
        for i,j in self.G.get_edges():
            qc.cx(i, j)
            qc.rz(gammas[count], j)
            qc.cx(i, j)
            count += 1
        return qc
    
    
    def x_mixer_operator(self, betas: List[float]) -> QuantumCircuit:
        """Create an instance of the x-mixer operator with angle 'beta'.
        
        Args:
            beta (list): Angles for the mixer operator.

        Returns:
            QuantumCircuit: Circuit representing the mixer operator.
        """
        qc = QuantumCircuit(self.N, self.N)
        count = 0
        for v in self.G.get_nodes():
            qc.rx(betas[count], v)
            count += 1
        return qc

    
    def get_circuit(self) -> QuantumCircuit:
        """Create an instance of the MaQaoa circuit with given parameters.
        
        Returns:
            QuantumCircuit: Circuit representing the MaQaoa.
        """
        qc = QuantumCircuit(self.N, self.N)
        params = ParameterVector("params", self.p * (self.N + self.M))
        qc.h(range(self.N))
        qc.barrier(range(self.N))
        left = 0
        middle = self.N
        right = self. N + self.M
        for layer in range(1,self.p+1):
            qc = qc.compose(self.cost_operator(params[middle : right]))
            qc.barrier(range(self.N))
            qc = qc.compose(self.x_mixer_operator(params[left : middle]))
            qc.barrier(range(self.N))
            left = right
            middle = left + self.N
            right = middle + self.M
        qc.barrier(range(self.N))
        if self.measure:  
            qc.measure(range(self.N), range(self.N))
            
        return qc
  