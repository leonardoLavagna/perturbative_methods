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
    axes[0].legend()
    axes[0].grid(True)
    # Right: Wavefunctions
    axes[1].plot(x, psi_0, label='Unperturbed $\\psi_n^{(0)}(x)$', color='blue')
    axes[1].plot(x, psi_0 + psi_1, label='1st Order $\\psi_n^{(0)}(x) + \\psi_n^{(1)}(x)$', color='orange')
    axes[1].plot(x, psi_total, label='2nd Order $\\psi_n^{(0)}(x) + \\psi_n^{(1)}(x) + \\psi_n^{(2)}(x)$', color='green')
    axes[1].set_title('Wavefunctions')
    axes[1].set_xlabel('Position $x$')
    axes[1].set_ylabel('Wavefunction $\\psi(x)$')
    axes[1].legend()
    axes[1].grid(True)

    st.pyplot(fig)

# Streamlit app layout
st.title("Potential well and its perturbations")
st.sidebar.header("Simulation Parameters")

# Sidebar inputs
L = st.sidebar.slider("Well Length (L)", 0.5, 5.0, 1.0)
epsilon = st.sidebar.slider("Perturbation Strength (epsilon)", 0.0, 25.0, 0.1)
n = st.sidebar.slider("Quantum Number (n)", 1, 5, 1)

st.header("Energy Levels and Wavefunctions")
plot_energy_and_wavefunctions(L, epsilon, n)

st.header("Wavefunction Evolution Animation")
frames = 100
t = np.linspace(0, 2 * np.pi, frames)
x = np.linspace(0, L, 200)

def init():
    line_real.set_data([], [])
    line_prob.set_data([], [])
    return line_real, line_prob

def animate(i):
    t_val = t[i]
    psi = Psi(n, x, L, t_val)
    psi_real = np.real(psi)
    psi_squared = np.abs(psi)**2
    line_real.set_data(x, psi_real)
    line_prob.set_data(x, psi_squared)
    return line_real, line_prob

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_xlim(0, L)
ax1.set_ylim(-2, 2)
ax2.set_xlim(0, L)
ax2.set_ylim(0, 4)
ax1.set_xlabel('Position (x)')
ax1.set_ylabel('Re[Ψ(x,t)]')
ax2.set_xlabel('Position (x)')
ax2.set_ylabel('|Ψ(x,t)|²')
line_real, = ax1.plot([], [], lw=2)
line_prob, = ax2.plot([], [], lw=2)

ani = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=True)

buf = BytesIO()
ani.save(buf, format='gif')
buf.seek(0)

st.image(buf, format='gif')
