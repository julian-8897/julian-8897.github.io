# Selected Projects

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

