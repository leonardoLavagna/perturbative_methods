#------------------------------------------------------------------------------
# config.py
#
# Configuration file
#
# © Leonardo Lavagna 2024
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------

# Size of the well
L = 1.0  

# Physical units (Plank's constand and mass)
hbar = 1.0  
m = 1.0  

# Line range and time range
x = np.linspace(0, L,  200)
t = np.linspace(0, 10, 200)

# Number of states and their coefficients (normalized)
n = 2
n_states = [1, 2]  
coeffs = [1, 1]  
coeffs = np.array(coeffs) / np.sqrt(np.sum(np.array(coeffs) ** 2))

# Maximum number of perturbed states to consider
max_states = 10  