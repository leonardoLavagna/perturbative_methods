import numpy as np
from config import *

def E_n(n, L):
    """Energy levels of a free particle in a box of length L."""
    return (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)

def f_n(n, x, L):
    """Unperturbed eigenfunction."""
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def Psi(n, x, L, t):
    """Time-dependent wavefunction for state n."""
    E = E_n(n, L)
    psi_x = f_n(n, x, L)
    return psi_x * np.exp(-1j * E * t / hbar)

def Psi_mixed(x, t, L, coeffs, n_states):
    """Superposition of eigenstates with coefficients coeffs."""
    psi_t = np.zeros_like(x, dtype=complex)
    for idx, n in enumerate(n_states):
        psi_x = f_n(n, x, L)
        E = E_n(n, L)
        time_factor = np.exp(-1j * E * t / hbar)
        psi_t += coeffs[idx] * psi_x * time_factor
    return psi_t
