Potential Bugs in the code:

----------------------------------

1. In training, use t_grid with higher order numerical integration (such as Simpson's rule) to compute the t-integral in the loss function.

2. There are infinite many equilibria of the gradient flow. Where the gradient flow converges depends on the initial condition. In order to make it consistent to the generation results, let's try to fix it to all zeros for now.

3. To track the convergence of the gradient flow, let's track the difference between the loss and the asymptotic loss. The asymptotic loss is computed using $A^*=\tilde{V}\tilde{U}^+$.
