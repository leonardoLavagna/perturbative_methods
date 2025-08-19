import numpy as np
import math
from config import *
from numpy.polynomial.hermite import hermval

def hermite_phys(n, z):
    """Physicists' Hermite polynomial H_n(z)."""
    c = np.zeros(n+1)
    c[-1] = 1.0
    return hermval(z, c)

def f_n(n, x, m, omega, hbar=hbar):
    """Normalized harmonic oscillator eigenfunction ψ_n(x)."""
    xi = np.sqrt(m*omega/hbar) * x
    norm = (m*omega/(np.pi*hbar))**0.25 / np.sqrt(2.0**n * math.factorial(n))
    return norm * hermite_phys(n, xi) * np.exp(-xi**2/2.0)

def E_n(n, m, omega, hbar=hbar):
    """Unperturbed energy of the HO."""
    return hbar*omega*(n+0.5)
