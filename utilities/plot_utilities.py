import numpy as np
import matplotlib.pyplot as plt
from config import *

def animate(i):
    t_val = t[i]
    psi = Psi(n, x, L, t_val)
    psi_real = np.real(psi)
    psi_squared = np.abs(psi)**2
    line_real.set_data(x, psi_real)
    line_prob.set_data(x, psi_squared)
    return line_real, line_prob

def animate_mixed(i):
    t_val = t[i]
    psi = Psi_mixed(x, t_val, L, coeffs, n_states)
    psi_real = np.real(psi)
    psi_squared = np.abs(psi)**2
    line_real.set_data(x, psi_real)
    line_prob.set_data(x, psi_squared)
    return line_real, line_prob


def plot_energy_and_wavefunctions(L=1.0, epsilon=0.1, n=1):
    x = np.linspace(0, L, 200)
    E0, E1, E2, psi_0, psi_1, psi_2, psi_total = energy_and_wavefunctions_corrections(x, L=L, epsilon=epsilon, n=n)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    #Left panel
    axes[0].hlines(E0, 0.5, 1.5, color='blue', label='Unperturbed $E_n^{(0)}$')
    axes[0].hlines(E0 + E1, 1.5, 2.5, color='orange', label='1st Order $E_n^{(0)} + E_n^{(1)}$')
    axes[0].hlines(E0 + E1 + E2, 2.5, 3.5, color='green', label='2nd Order $E_n^{(0)} + E_n^{(1)} + E_n^{(2)}$')
    axes[0].set_title('Energy Levels')
    axes[0].set_xticks([])
    axes[0].set_ylabel('Energy')
    axes[0].legend()
    axes[0].grid(True)
    #Right panel
    axes[1].plot(x, psi_0, label='Unperturbed $\psi_n^{(0)}(x)$', color='blue')
    axes[1].plot(x, psi_0 + psi_1, label='1st Order $\psi_n^{(0)}(x) + \psi_n^{(1)}(x)$', color='orange')
    axes[1].plot(x, psi_total, label='2nd Order $\psi_n^{(0)}(x) + \psi_n^{(1)}(x) + \psi_n^{(2)}(x)$', color='green')
    axes[1].set_title('Wavefunctions')
    axes[1].set_xlabel('Position $x$')
    axes[1].set_ylabel('Wavefunction $\psi(x)$')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()
