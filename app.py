import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utilities.perturbed_potential_well_utilities import *
from utilities.unperturbed_potential_well_utilities import *
from config import *

st.title("Perturbative methods in action")

# Select problem
def select_problem():
    return st.sidebar.selectbox("Select a problem", ["Potential Well"], index=0)

problem = select_problem()

if problem == "Potential Well":
    st.header("Potential Well Simulation")
    
    # Sidebar controls
    L = st.sidebar.slider("Well Length (L)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    epsilon = st.sidebar.slider("Perturbation Strength (ε)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    n = st.sidebar.slider("Quantum Number (n)", min_value=1, max_value=5, value=1, step=1)
    
    # Compute energy levels and wavefunctions
    x = np.linspace(0, L, 200)
    E0, E1, E2, psi_0, psi_1, psi_2, psi_total = energy_and_wavefunctions_corrections(x, L=L, epsilon=epsilon, n=n)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Energy Levels Plot
    axes[0].hlines(E0, 0.5, 1.5, color='blue', label='Unperturbed $E_n^{(0)}$')
    axes[0].hlines(E0 + E1, 1.5, 2.5, color='orange', label='1st Order $E_n^{(0)} + E_n^{(1)}$')
    axes[0].hlines(E0 + E1 + E2, 2.5, 3.5, color='green', label='2nd Order $E_n^{(0)} + E_n^{(1)} + E_n^{(2)}$')
    axes[0].set_title('Energy Levels')
    axes[0].set_xticks([])
    axes[0].set_ylabel('Energy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Wavefunctions Plot
    axes[1].plot(x, psi_0, label='Unperturbed $\\psi_n^{(0)}(x)$', color='blue')
    axes[1].plot(x, psi_0 + psi_1, label='1st Order $\\psi_n^{(0)}(x) + \\psi_n^{(1)}(x)$', color='orange')
    axes[1].plot(x, psi_total, label='2nd Order $\\psi_n^{(0)}(x) + \\psi_n^{(1)}(x) + \\psi_n^{(2)}(x)$', color='green')
    axes[1].set_title('Wavefunctions')
    axes[1].set_xlabel('Position $x$')
    axes[1].set_ylabel('Wavefunction $\\psi(x)$')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
