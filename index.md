# Selected Projects

## Continuous-Time Modelling of Black Hole Binary Evolution with Neural ODEs

**Technologies:** PyTorch, Neural ODEs, torchdiffeq, AdamW, Curriculum Learning

**📄 Published:** [Monthly Notices of the Royal Astronomical Society (MNRAS)](https://doi.org/10.1093/mnras/stag135) | [arXiv:2601.13019](https://arxiv.org/abs/2601.13019)

### Overview

Developed parameterised neural ordinary differential equations (PNODEs) as surrogate models for supermassive black hole binary dynamics in galaxy mergers. The approach provides continuous-time predictions of orbital evolution across a two-dimensional parameter space, achieving significant computational speedup (weeks to seconds) while maintaining median prediction errors around 1% on held-out test data.

### Problem & Approach

- Trained parameter-conditioned neural ODEs to learn the dynamical system governing coupled evolution of orbital energy and angular momentum from 156 N-body simulations
- Addressed challenges in learning smooth trajectories from data with inherent stochastic noise due to gravitational three-body interactions
- Built surrogate models to enable efficient parameter space exploration for astrophysics applications where direct simulation is computationally prohibitive

### Technical Implementation

- Designed parameter-conditioned neural ODE architecture that takes simulation parameters as inputs alongside state variables, enabling a single model to generalize across continuous parameter space
- Implemented two-stage curriculum learning: initial per-trajectory training followed by joint ensemble training to capture parameter-dependent dynamics
- Engineered weighted Huber loss function with differential weighting (5× on angular momentum) to improve coupled variable predictions through implicit dependencies
- Integrated adaptive ODE solver (Dormand-Prince) for numerically stable long-horizon trajectory integration
- Applied domain-informed preprocessing: log-transformation for scale invariance, input normalization, and careful handling of phase transitions in the dynamics

### Results

- Achieved median fractional errors ~1% on both target variables across 24 held-out test trajectories within the training parameter distribution
- Demonstrated interpolation capability across the two-dimensional parameter space with a single unified model
- Observed some extrapolation to higher-resolution simulations beyond the training distribution, though with increased uncertainty
- Validated that predictions maintain physical consistency through downstream calculations of derived quantities (orbital elements, timescales)
- Reduced inference cost by multiple orders of magnitude compared to full simulation while capturing key dynamical trends

---------

## Inverse Burgers Equation Solver with Cross-Framework PINNs

**Technologies:** JAX, PyTorch, Tesseract, Equinox, Docker, Streamlit

**🏆 Honorable Mention - [Tesseract Hackathon 2025](https://pasteurlabs.ai/insights/tesseract-hackathon-winners)** (Pasteur Labs & ISI)

### Overview

Developed a backend-agnostic physics-informed neural network (PINN) system for solving inverse problems in fluid dynamics. The project demonstrates pipeline-level automatic differentiation across deep learning frameworks, enabling JAX optimizers to compute gradients through PyTorch models via the Tesseract framework.

### Problem & Approach

- Implemented an inverse solver for the 1D viscous Burgers equation to infer unknown viscosity parameters from noisy observational data
- Designed a PINN architecture with Fourier feature encoding to mitigate spectral bias, processing spatial-temporal inputs through a 130→64→64→64→1 MLP
- Minimized a composite loss function incorporating data fidelity, PDE residuals, initial conditions, and boundary conditions

### Technical Implementation

- Built dual backend implementations (JAX/Equinox and PyTorch) with identical optimization pipelines, packaged as Docker containers
- Implemented Tesseract's differentiable programming interface (`apply`, `vector_jacobian_product`, `jacobian_vector_product`) for both backends to enable cross-framework gradient computation
- Leveraged native automatic differentiation (`jax.grad`, `torch.autograd.grad`) for computing solution derivatives (∂u/∂x, ∂u/∂t, ∂²u/∂x²)
- Created an interactive Streamlit application with real-time training visualization, hyperparameter tuning, and gradient flow inspection

### Results

- Achieved accurate viscosity inference (ν = 0.05) with both backends converging after 100 epochs
- Demonstrated framework-agnostic scientific ML pipeline where backend selection only affects internal autograd implementation, not system-level code
- Validated approach on synthetic data with realistic noise levels (σ = 0.02)
- Recognized as one of three honorable mentions among submissions from Johns Hopkins University, industry researchers, and international teams

**Repository:** [github.com/julian-8897/tesseract-pinn-inverse-burgers](https://github.com/julian-8897/tesseract-pinn-inverse-burgers)

