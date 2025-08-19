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

def select_problem():
    return st.sidebar.selectbox("Select a topic", ["Introduction", "The potential well problem"], index=0)
    
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
    express energy corrections as a power series in $ \varepsilon$, and approximate the perturbed wavefunction, starting from a known unperturbed solution, so that we can laverage an 
    approximation scheme that allows us to extend the knowledge about an exactly solvable system to other quantum systems, at least locally. In particular, the 
    goal of perturbation theory is to extend the unperturbed solution $f_n$ to the perturbed case and reconstruct the overall perturbed wavefunction $\Psi'$ from $f_n'$ and the corresponding 
    time evolution function. This schemes yield separation of scales (depending on the size of $\varepsilon$) and corrections to progressively approximate even the most difficult cases.
    

    In this app, we provide the solution of some one-dimensional quantum mechanical problems with insightful visualizations.
    ''')
    

################################################
# 3. DIRECT PERTURBATIVE METHOD
################################################  
elif problem == "Perturbative methods":
    st.header("Perturbative methods")
    st.markdown(r"""
    A direct perturbative method can be applied to solve the energy-eigenvalue equation in the time-independent case, when a small deviation from the unperturbed 
    problem is considered. In particular, at the first order, the potential changes as $V(x)\to V'(x; \varepsilon)=V(x)+\varepsilon V^{(1)}(x)$, 
    and thus the Hamiltonian becomes $\hat{H}\to \hat{H}'=\hat{H}+\varepsilon \hat{H}^{(1)}$, for given $V^{(1)}$, $\hat{H}^{(1)}$ suitable perturbation terms. 
    We will see that in this setting it is easy to find approximate solutions of the perturbed case using the unperturbed solution with a power series expansion 
    trick, provided that different energy levels in the unperturbed spectrum have nonzero gap (i.e., $E_n-E_m\ne 0$ for every $n\ne m$). 
    The key idea of this method can also be extended to higher-order deviations 
    $V\to V'=V+\varepsilon V^{(1)}+\varepsilon^2V^{(2)}+\dots$ and $\hat{H}\to \hat{H}'=\hat{H}+\varepsilon\hat{H}^{(1)}+\varepsilon^2\hat{H}^{(2)}+\dots$.
    Let's start with the general case where there are at most countable energy levels. Expanding the energy level $E_n$ and the solution $f_n$ of 
    the unperturbed problem in a power series in terms of $\varepsilon$, and considering corrections $f_n^{(k)}, E_n^{(k)}$ of order $k$ to the original solution and energy, yields
    $$
    (\hat{H}+\varepsilon \hat{H}')\Bigl(\sum_{k=0}^\infty \varepsilon^k{f}_n^{(k)}\Bigr)=\Bigl(\sum_{k=0}^\infty\varepsilon^k {E}_n^{(k)} \Bigr)\Bigl(\sum_{k=0}^\infty \varepsilon^k{f}_n^{(k)}\Bigr)\,,
    $$
    where $f_n^{(0)}:=f_n$ and $E_n^{(0)}:=E_n$ by definition. Collecting the terms which are powers of $\varepsilon$ and after simple algebraic manipulations,
    we get the first-order correction formula:
    $$
    E_n^{(1)}=\langle f_n|\hat{H}'f_n\rangle\,,
    $$
    which shows that the first energy shift is the expectation value of the perturbation calculated in the unperturbed state. With similar calculations, 
    it is also easy to get the second-order correction to the energy, which (in the non-degenerate case) reads:
    $$
    E_n^{(2)}=\sum_{h\ne k}\frac{|\langle f_h | \hat{H}' f_k\rangle|^2}{E_h-E_k}\,.
    $$
    """)

    
################################################
# 3. POTENTIAL WELL
################################################  
elif problem == "The potential well problem":
    st.header("The potential well problem")
    st.markdown(r'''
    Consider a particle of mass $m$ confined to an interval $I := [0, L]$ of length $L > 0$. The potential is given by
    $$
    V(x) = \begin{cases} 0, & x \in I, \\ +\infty, & \text{otherwise}, \end{cases}
    $$
    which means the particle experiences infinite forces at $x = 0$ and $x = L$, preventing it from escaping.
    Outside the well, the wavefunction is zero, $f(x) = 0$ for $x \notin I$, since the probability of finding the particle there is zero. 
    Inside the well, the Schrödinger equation simplifies to
    $$
    \frac{d^2f}{dx^2} = - k^2 f,
    $$
    where
    $$
    k := \hbar^{-1} \sqrt{2mE}.
    $$
    This is the equation of a [harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator), with the general solution
    $$
    f(x) = A \sin(kx) + B \cos(kx).
    $$
    Applying the boundary condition $f(0) = 0$ gives $B = 0$, so $f(x) = A \sin(kx)$. To avoid the trivial solution ($A = 0$), the boundary condition $f(L) = 0$ requires
    $$
    kL = n\pi, \quad n \in \mathbb{Z}_{>0}.
    $$
    Thus, the allowed wave numbers are
    $$
    k(n) = \frac{n\pi}{L}.
    $$
    By normalization, we find $A = \sqrt{\frac{2}{L}}$, leading to the eigenfunctions
    $$
    f_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi}{L}x\right).
    $$
    These functions form a complete, orthonormal basis, allowing the general solution for a potential well to be written as
    $$
    \Psi(x,t) = \sum_{n=1}^\infty c_n \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi}{L}x\right) e^{-iE_nt/\hbar}.
    $$
    The coefficients $c_n$ are determined from the initial condition:
    $$
    \Psi(x,0) = \sum_{n=1}^\infty c_n f_n(x).
    $$
    Multiplying both sides by $f_m^*(x)$, integrating, and using orthonormality gives
    $$
    c_n = \sqrt{\frac{2}{L}} \int_0^L dx \sin\left(\frac{n\pi}{L}x\right) \Psi(x,0).
    $$
    ''')
    
    st.subheader("Energy shifts and perturbed wavefunctions")
    st.markdown(r'''
    If we consider the perturbation $V'=V_0$ obtained by adding a positive constant $V_0>0$ to the zero potential in the interval $I=[0,L]$,
    we have the first-order correction of the energy:
    $$
    E_n^{(1)}=\langle f_n |V_0 f_n\rangle =V_0\,.
    $$
    To find the first-order correction to the wave function, we need to solve:
    $$
    (\hat{H}-E_n)f_n^{(1)}=-(\hat{H}^{(1)}-E_n^{(1)})f_n\,.
    $$
    Since the unperturbed wave functions constitute a complete set of orthonormal functions, we can write:
    $$
    f_n^{(1)}=\sum_{m\ne n}c_{mn}f_m
    $$
    for suitable constants $c_{mn}$, which means that:
    $$
    \sum_{m\ne n} (E_m-E_n)c_{mn}f_m=-(\hat{H}^{(1)}-E^{(1)}_n)f_n
    $$
    and taking the inner product with $f_k$ yields:
    $$
    \sum_{m\ne n}(E_m-E_n)c_{mn}\langle f_k|f_m\rangle = -\langle f_k|\hat{H}^{(1)} f_n\rangle + E_n^{(1)}\langle f_k|f_n\rangle\,,
    $$
    or, equivalently, 
    $$
    c_{mn}=\frac{\langle f_m|\hat{H}^{(1)} f_n\rangle}{E_n-E_m}\,.
    $$
    Putting everything together, we have the general first-order correction to the wave function:
    $$
    f_n^{(1)}=\sum_{m\ne n} \frac{\langle f_m|\hat{H}^{(1)} f_n\rangle}{E_n-E_m}f_m\,,
    $$
    which, in the potential well case we are considering, with a constant perturbation $V_0$, means:
    $$
    f_n^{(1)}=\sum_{m\ne n} \frac{\langle f_m|V_0f_n\rangle}{E_n-E_m}f_m=V_0\sum_{m\ne n}\frac{f_m}{E_n-E_m}\,.
    $$
    Similar calculations can be carried out to get the second-order corrections, and are quite general (i.e., not limited to the case of the potential well, but applicable, 
    for example, in the harmonic oscillator case). 
    Clearly, if $E_m-E_k=0$, the previous derivation of the closed formula expression for the first-order approximation of the wave function is not valid, 
    and the solution must be deduced using degenerate perturbation theory. In that case, indirect perturbative methods could also be tried as a valid alternative. 
    ''')
    st.subheader("Visualizations")
    st.markdown(r'''
    Here we show, in the left panel, the difference between energy levels and wavefunctions for the unperturbed energy
    $$
    E_n^{(0)}=\frac{(n\pi\hbar)^2}{2mL^2}\,,
    $$
    the first order correction $E_n^{(0)}+E^{(1)}_n$, where $E_n^{(1)}$ is obtained as the expectation value of the perturbation calculated in the
    unperturbed state, and the second order correction $E_n^{(0)}+E^{(1)}_n + E^{(2)}_n$, where 
    $$
    E_n^{(2)}= \sum_{m\ne n}\frac{|\langle f_m^{(0)} \ | \ V'f_n^{(0)}|^2}{E_n^{(0)}-E_{m}^(0)}\,.
    $$
    In the right panel we see the corrections of the wavefunctions, starting from the unperturbed case 
    $$
    f_n^{(0)}(x)=\sqrt{\frac{2}{L}}\sin(\frac{n\pi x}{L})\,,
    $$
    then the first order correction $f_n^{(0)}(x)+f_n^{(1)}(x)$, where 
    $$
    f_n^{(1)}(x)=\sum_{m\ne n}\frac{|\langle f_m^{(0)} \ | \ V'f_n^{(0)}|^2}{E_n^{(0)}-E_{m}^(0)} f_n^{(0)}(x)
    $$
    and, finally, the second order correction $f_n^{(0)}(x)+f_n^{(1)}(x) + f_n^{(2)}(x)$, where
    $$
    f_n^{(2)}(x)=\sum_{m\ne n}\sum_{k\ne m}\frac{|\langle f_k^{(0)} \ | \ V'f_m^{(0)}|^2|\langle f_m^{(0)} \ | \ V'f_n^{(0)}|^2}{(E_n^{(0)}-E_{m}^(0))(E_m^{(0)}-E_{k}^(0))} f_k^{(0)}(x)\,.
    $$

    With the options in the menu on the left side of this page, we see that increasing $\epsilon$ increases the influence of the perturbing potential, making corrections to the energy levels and wavefunctions more pronounced. 
    Moreover, higher $n$-states exhibit larger deviations due to their higher energy levels and larger overlaps with the perturbing potential.
    ''')
    L = st.sidebar.slider("Well Length (L)", min_value=0.5, max_value=10.0, value=1.0, step=0.1)
    epsilon = st.sidebar.slider("Perturbation Strength (ε)", min_value=0.0, max_value=10.0, value=0.1, step=0.1)
    n = st.sidebar.slider("Quantum Number (n)", min_value=1, max_value=10, value=1, step=1)
    # Compute energy levels and wavefunctions
    x = np.linspace(0, L, 200)
    E0, E1, E2, psi_0, psi_1, psi_2, psi_total = energy_and_wavefunctions_corrections(x, L=L, epsilon=epsilon, n=n)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes[0].hlines(E0, 0.5, 1.5, color='blue', label='Unperturbed $E_n^{(0)}$')
    axes[0].hlines(E0 + E1, 1.5, 2.5, color='red', label='1st Order $E_n^{(0)} + E_n^{(1)}$')
    axes[0].hlines(E0 + E1 + E2, 2.5, 3.5, color='green', label='2nd Order $E_n^{(0)} + E_n^{(1)} + E_n^{(2)}$')
    axes[0].set_title('Energy Levels')
    axes[0].set_xticks([])
    axes[0].set_ylabel('Energy')
    axes[0].grid(True)
    axes[1].plot(x, psi_0, label='Unperturbed $\\psi_n^{(0)}(x)$', color='blue')
    axes[1].plot(x, psi_0 + psi_1, label='1st Order $\\psi_n^{(0)}(x) + \\psi_n^{(1)}(x)$', color='red')
    axes[1].plot(x, psi_total, label='2nd Order $\\psi_n^{(0)}(x) + \\psi_n^{(1)}(x) + \\psi_n^{(2)}(x)$', color='green')
    axes[1].set_title('Wavefunctions')
    axes[1].set_xlabel('Position $x$')
    axes[1].set_ylabel('Wavefunction $\\psi(x)$')
    axes[1].grid(True)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.tight_layout()
    st.pyplot(fig)
