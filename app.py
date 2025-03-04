import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utilities.perturbed_potential_well_utilities import *
from utilities.unperturbed_potential_well_utilities import *
from config import *


################################################
# COLOPHON
################################################   
st.title("Perturbative methods in action")

# Select problem
def select_problem():
    return st.sidebar.selectbox("Select a problem", ["Introduction", "Potential Well"], index=0)

problem = select_problem()

################################################
# 1. INTRODUCTION
################################################  
if problem == "Introduction":
    st.markdown(r'''
    In [quantum mechanics](https://en.wikipedia.org/wiki/Quantum_mechanics), 
    instead of tracking a particle's motion as $x = x(t)$ following [Newton's equation](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion), we determine its [wavefunction](https://en.wikipedia.org/wiki/Wave_function) $\Psi = \Psi(x, t)$, obtained by solving the [Schrödinger's equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation):
    $$
    i\hbar \frac{\partial \Psi}{\partial t} = -\frac{\hbar^2}{2m} \frac{\partial^2 \Psi}{\partial x^2} + V\Psi,
    $$
    where $i^2 = -1$, and $\hbar = 1.054573 \times 10^{-34}$Js is the reduced Planck constant. 
    The wavefunction $\Psi$ is complex-valued, and its squared modulus represents the probability density of finding the particle at a 
    given position (cf. the [Copenhagen interpretation](https://en.wikipedia.org/wiki/Copenhagen_interpretation)). To ensure a proper probability distribution, it must satisfy the normalization condition:
    $$
    \int_{-\infty}^{+\infty} dx |\Psi(x,t)|^2 = 1.
    $$
    Schrödinger's equation can often be solved using separation of variables by assuming:
    $$
    \Psi(x,t) = f(x)g(t),
    $$
    where $f:\mathbb{R} \to \mathbb{R}$ and $g:\mathbb{R} \to \mathbb{C}$ are smooth functions, each satisfying the normalization condition. 
    The function $f(x)$, often referred to as the wavefunction itself, determines the probability density, as the probability of finding the particle in the region $[x, x + \Delta x]$ at time $t$ is proportional to $|f(x)|^2 \Delta x$. Substituting, we obtain:
    $$
    i\hbar f \frac{dg}{dt} = -\frac{\hbar^2}{2m} \frac{d^2 f}{dx^2} g + Vfg.
    $$
    Dividing by $fg$ and introducing the separation constant $E$, interpreted as the system's energy, we arrive at two ordinary differential equations:
    $$
    \frac{dg}{dt} = -\frac{iE}{\hbar} g,
    $$
    and
    $$
    -\frac{\hbar^2}{2m} \frac{d^2 f}{dx^2} + Vf = Ef.
    $$
    The second equation is an eigenvalue problem for the [Hamiltonian operator](https://en.wikipedia.org/wiki/Hamiltonian_(quantum_mechanics)):
    $$
    \hat{H} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x).
    $$
    Thus, the energy eigenvalue equation is:
    $$
    \hat{H} f = E f.
    $$
    For nontrivial $V(x)$, we generally have a discrete set of solutions $\{ f_n \}$ corresponding to energy levels $\{ E_n \}$:
    $$
    \hat{H} f_n = E_n f_n.
    $$
    Moreover, solving for $g$, we have:
    $$
    g(t) = e^{-\frac{iE t}{\hbar}}.
    $$
    Thus, the eigenstate solution is:
    $$
    \Psi_n(x,t) = c_n f_n(x) e^{-\frac{iE_n t}{\hbar}},
    $$
    where $c_n$ is an integration constant. By the [superposition principle](https://en.wikipedia.org/wiki/Quantum_superposition), the general solution of the Schrödinger equation is:
    $$
    \Psi(x,t) = \sum_{n=1}^{\infty} c_n f_n(x) e^{-\frac{iE_n t}{\hbar}}.
    $$
    The key challenge is solving the time-independent Schrödinger equation and determining the coefficients $c_n$ to match initial conditions. 
    In [perturbation theory](https://en.wikipedia.org/wiki/Perturbation_theory_(quantum_mechanics)), we introduce a small perturbation $\hat{H}'(\varepsilon)$, 
    express energy corrections as a power series in $ \varepsilon$, and approximate the perturbed wavefunction. 
    The goal is to extend the unperturbed solution $f_n$ to the perturbed case and reconstruct the overall perturbed wavefunction $\Psi'$ from $f_n'$ and the corresponding 
    time evolution function.

    In this app, we provide the solution of some one-dimensional quantum mechanical problems with insightful visualizations.
    ''')

################################################
# 2. POTENTIAL WELL
################################################  
elif problem == "Potential Well":
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
