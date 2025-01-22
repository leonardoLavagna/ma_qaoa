#------------------------------------------------------------------------------
# SETUP FOR QUANTUM OPTIMIZATION ROUTINES (MA-QAOA)
# @ Leonardo Lavagna 2024
#------------------------------------------------------------------------------


from qiskit_aer import AerSimulator


#p (int): The number of qaoa layers.
p = 1

#idx (int): An index between 0 and 15 that allows to select a graph instance from the data directory.
idx = 0

#seed (int): A seed for the pseudorandom generators (needed for reproducibility).
seed = 121

#verbose (bool): A boolean variable that allows to work in debugging mode if True.
verbose = False

#backend (Qiskit backend): A Qiskit backend to simulate a quantum device.
backend = AerSimulator(method='automatic')

#mixer (str): Type of qaoa mixer operator. Currently are available 'x', 'y', 'xx', 'xy' and 'yy' mixers.
mixer = "x" 

#method (str): Type of optimization method to be used with `scipy.optimize.minimize`. Available 'BFGS', 'L-BFGS-B', 'COBYLA' and 'SLSQP'.
method = 'COBYLA' 