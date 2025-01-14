import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.animation import FuncAnimation
from io import BytesIO
from utilities.perturbed_potential_well_utilities import *
from utilities.unperturbed_potential_well_utilities import *

def plot_energy_and_wavefunctions(L, epsilon, n):
    x = np.linspace(0, L, 200)
    E0, E1, E2, psi_0, psi_1, psi_2, psi_total = energy_and_wavefunctions_corrections(x, L, epsilon, n)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Left: Energy levels
    axes[0].hlines(E0, 0.5, 1.5, color='blue', label='Unperturbed $E_n^{(0)}$')
    axes[0].hlines(E0 + E1, 1.5, 2.5, color='orange', label='1st Order $E_n^{(0)} + E_n^{(1)}$')
    axes[0].hlines(E0 + E1 + E2, 2.5, 3.5, color='green', label='2nd Order $E_n^{(0)} + E_n^{(1)} + E_n^{(2)}$')
    axes[0].set_title('Energy Levels')
    axes[0].set_xticks([])
    axes[0].set_ylabel('Energy')
    #axes[0].legend()
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    axes[0].grid(True)
    # Right: Wavefunctions
    axes[1].plot(x, psi_0, label='Unperturbed $\\psi_n^{(0)}(x)$', color='blue')
    axes[1].plot(x, psi_0 + psi_1, label='1st Order $\\psi_n^{(0)}(x) + \\psi_n^{(1)}(x)$', color='orange')
    axes[1].plot(x, psi_total, label='2nd Order $\\psi_n^{(0)}(x) + \\psi_n^{(1)}(x) + \\psi_n^{(2)}(x)$', color='green')
    axes[1].set_title('Wavefunctions')
    #axes[1].set_xlabel('Position $x$')
    axes[1].set_ylabel('Wavefunction $\\psi(x)$')
    #axes[1].legend()
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    axes[1].grid(True)
    st.pyplot(fig)

# Streamlit app layout
st.title("Potential well and its perturbations")
st.sidebar.header("Simulation Parameters")

# Sidebar inputs
L = st.sidebar.slider("Well Length (L)", 1, 10, 1)
epsilon = st.sidebar.slider("Perturbation Strength (epsilon)", 0.0, 1.0, 0.01)
n = st.sidebar.slider("Quantum Number (n)", 1, 10, 1)

st.header("Energy Levels and Wavefunctions")
plot_energy_and_wavefunctions(L, epsilon, n)
