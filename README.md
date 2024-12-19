# Autodifferentiable ODEs in MATLAB

This repository demonstrates how to use MATLAB’s Deep Learning Toolbox for computing gradients of ODE solutions (and losses related to them). By leveraging automatic differentiation, we can easily find sensitivities with respect to parameters, initial conditions, and other inputs; this enables rapid prototyping of gradient-based optimization of ODE models with respect to data.

While we leverage MATLAB's deep learning toolbox, this demonstration does not involve any neural networks. MATLAB’s deep learning toolbox is not limited to neural networks, and allows for generic computations with automatic differentiation; of course, neural networks are also supported.
We use this generic autodifferentiation framework to our advantage to compute gradients of ODE solutions with respect to model inputs; in the past, this often required manual derivations or finite differences.
This is particularly useful for gradient-based optimization/inference and sensitivity analysis. 

**Key Features:**
- **Automatic Differentiation:** Use `dlgradient` and `dlfeval` to compute gradients without manually deriving expressions.
- **Parameter & Initial Condition Sensitivity:** Evaluate how ODE solutions change with variations in model inputs.

**Requirements:**
- MATLAB R2023b (tested)
- Deep Learning Toolbox

**Usage:**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/autodiff_ODE_matlab.git
   ```
2. Open MATLAB and navigate to the repository folder
3. Run the demo: `autodiff_ODE_demo.m`
- This script demonstrates how to compute gradients of ODE solutions (Lorenz '63) with respect to parameters and initial conditions.
- It makes plots to visualize the sensitivities.

**Other Resources:**
- [MATLAB's Neural ODE demo](https://www.mathworks.com/help/deeplearning/ug/dynamical-system-modeling-using-neural-ode.html)
- [`dlode45` documentation](https://www.mathworks.com/help/deeplearning/ref/dlarray.dlode45.html)
- [`dlarray` documentation](https://www.mathworks.com/help/deeplearning/ref/dlarray.html)
- [`dlfeval` documentation](https://www.mathworks.com/help/deeplearning/ref/dlfeval.html)
- [`dlgradient` documentation](https://www.mathworks.com/help/deeplearning/ref/dlarray.dlgradient.html)
- If you are more broadly interested in fitting mechanistic/machine-learnt dynamical models to real-world (i.e., noisy, partially-observed, irregularly-sampled) data, check out [CD-Dynamax, a JAX-based toolbox](https://github.com/hd-UQ/cd_dynamax).

**Contributing:**
- Feel free to open an issue or submit a pull request if you have any suggestions or improvements.
- If you find this repository helpful, please consider starring it!
- If you use this code in your research, please cite this repository.

