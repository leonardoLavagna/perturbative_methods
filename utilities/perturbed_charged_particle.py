import numpy as np
from scipy.integrate import quad
from config import *
from utilities.unperturbed_charged_particle import *

def V_prime(x, q, E_field):
    """Perturbing potential: uniform electric field."""
    return -q * E_field * x

def H_prime_mn(m, n, q, E_field, L):
    """Matrix element of the perturbation between states m and n."""
    def integrand(x):
        return f_n(n, x, L) * V_prime(x, q, E_field) * f_n(m, x, L)
    result, _ = quad(integrand, 0, L)
    return result

def first_order_correction(n, q, E_field, L):
    """First-order correction (expected to vanish by symmetry)."""
    return H_prime_mn(n, n, q, E_field, L)

def Psi_1_prime(x, n, q, E_field, L):
    """First-order correction to the wavefunction."""
    correction = np.zeros_like(x)
    E_n0 = E_n(n,L)
    for m in range(1, max_states + 1):  
        if m == n:
            continue
        E_m0 = E_n(m, L)
        coeff = H_prime_mn(m, n, q, E_field, L) / (E_n0 - E_m0)
        correction += coeff * f_n(m, x, L)
    return correction

def second_order_correction(n, q, E_field, L):
    """Second-order correction to the energy."""
    correction = 0
    E_n0 = E_n(n, L)
    for m in range(1, max_states + 1):
        if m == n:
            continue
        H_prime_mn_value = H_prime_mn(m, n, q, E_field, L)
        correction += (abs(H_prime_mn_value)**2) / (E_n0 - E_n(m,L))
    return correction

def energy_and_wavefunctions_corrections(x, L, n=1, q=1.0, E_field=0.1):
    """Compute unperturbed and perturbed energies/wavefunctions."""
    psi_0 = f_n(n, x, L)  
    psi_1 = Psi_1_prime(x, n, q, E_field, L)  
    psi_total = psi_0 + psi_1  
    psi_total /= np.sqrt(np.trapz(psi_total**2, x))
    E0 = E_n(n, L)
    E1 = first_order_correction(n, q, E_field, L)
    E2 = second_order_correction(n, q, E_field, L)
    return E0, E1, E2, psi_0, psi_1, psi_total
