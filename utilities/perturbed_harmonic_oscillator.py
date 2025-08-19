import numpy as np
import math
from config import *
from numpy.polynomial.hermite import hermval

def first_order_correction(n, epsilon, m, omega, hbar=hbar):
    """First-order correction to the energy (vanishes for V'=εx)."""
    return 0.0

def matrix_element_x(n, m_, m, omega, hbar=hbar):
    """<m|x|n> analytic formula for HO."""
    if m_ == n+1:
        return np.sqrt(hbar/(2*m*omega)) * np.sqrt(n+1)
    elif m_ == n-1:
        return np.sqrt(hbar/(2*m*omega)) * np.sqrt(n)
    else:
        return 0.0

def second_order_correction(n, epsilon, m, omega, hbar=hbar):
    """Second-order energy correction (constant, independent of n)."""
    # From analytic derivation: ΔE_n^(2) = -ε^2 / (2 m ω^2)
    return - (epsilon**2) / (2.0*m*omega**2)

def Psi_1_prime(x, n, epsilon, m, omega, hbar=hbar):
    """First-order correction to wavefunction for V'=εx (mixes n±1)."""
    E0 = E_n(n, m, omega, hbar)
    correction = np.zeros_like(x, dtype=float)

    # n+1 contribution
    denom_up = E0 - E_n(n+1, m, omega, hbar)   # = -ħω
    coeff_up = (epsilon * matrix_element_x(n, n+1, m, omega, hbar)) / denom_up
    correction += coeff_up * f_n(n+1, x, m, omega, hbar)

    # n-1 contribution
    if n > 0:
        denom_dn = E0 - E_n(n-1, m, omega, hbar)   # = +ħω
        coeff_dn = (epsilon * matrix_element_x(n, n-1, m, omega, hbar)) / denom_dn
        correction += coeff_dn * f_n(n-1, x, m, omega, hbar)

    return correction

def energy_and_wavefunctions_corrections(x, n, epsilon, m, omega, hbar=hbar):
    """Return E0, E1, E2 and ψ^(0), ψ^(1), ψ^(0)+ψ^(1) for plotting."""
    psi_0 = f_n(n, x, m, omega, hbar)
    psi_1 = Psi_1_prime(x, n, epsilon, m, omega, hbar)
    psi_total = psi_0 + psi_1
    # Normalize corrected wavefunction
    norm = np.sqrt(np.trapz(np.abs(psi_total)**2, x))
    if norm > 0:
        psi_total /= norm

    E0 = E_n(n, m, omega, hbar)
    E1 = first_order_correction(n, epsilon, m, omega, hbar)
    E2 = second_order_correction(n, epsilon, m, omega, hbar)

    return E0, E1, E2, psi_0, psi_1, psi_total

