import numpy as np
from scipy.integrate import quad
from config import *
from utilities.unperturbed_potential_well_utilities import *

def V_prime(x, epsilon):
    return epsilon * x**2

def H_prime_mn(m, n, epsilon, L):
    def integrand(x):
        return f_n(x, n, L) * V_prime(x, epsilon) * f_n(x, m, L)
    result, _ = quad(integrand, 0, L)
    return result

def first_order_correction(n, epsilon, L):
    return H_prime_mn(n, n, epsilon, L)

def Psi_1_prime(x, n, epsilon, L):
    correction = np.zeros_like(x)
    E_n0 = E_n(n,L)
    for m in range(1, max_states + 1):  
        if m == n:
            continue
        E_m0 = E_n(m, L)
        coeff = H_prime_mn(m, n, epsilon, L) / (E_n0 - E_m0)
        correction += coeff * f_n(x, m, L)
    return correction

def second_order_correction(n, epsilon, L):
    correction = 0
    E_n0 = E_n(n, L)
    for m in range(1, max_states + 1):
        if m == n:
            continue
        H_prime_mn_value = H_prime_mn(m, n, epsilon, L)
        correction += (abs(H_prime_mn_value)**2) / (E_n0 - E_n(m,L))
    return correction

def Psi_2_prime(x, n, epsilon, L):
    correction = np.zeros_like(x)
    E_n0 = E_n(n, L)
    for m in range(1, max_states + 1): 
        if m == n:
            continue
        E_m0 = E_n(m, L)
        coeff_mn = H_prime_mn(m, n, epsilon, L) / (E_n0 - E_m0)
        for k in range(1, max_states + 1):  
            if k == m:
                continue
            E_k0 = E_n(k, L)
            coeff_mk = H_prime_mn(k, m, epsilon, L) / (E_m0 - E_k0)
            correction += coeff_mn * coeff_mk * f_n(x, k, L)
    return correction

def energy_and_wavefunctions_corrections(x, L, epsilon=0.1, n=1):
    psi_0 = f_n(x, n, L)  
    psi_1 = Psi_1_prime(x, n, epsilon, L)  
    psi_2 = Psi_2_prime(x, n, epsilon, L)  
    psi_total = psi_0 + psi_1 + psi_2  
    psi_total /= np.sqrt(np.trapz(psi_total**2, x))
    E0 = E_n(n, L)
    E1 = first_order_correction(n, epsilon, L)
    E2 = second_order_correction(n, epsilon, L)
    return E0, E1, E2, psi_0, psi_1, psi_2, psi_total
