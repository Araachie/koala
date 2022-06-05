### Abstract

Optimization is often cast as a deterministic problem, where the solution is found through some iterative procedure such as gradient descent. However, when training neural networks the loss function changes over (iteration) time due to the randomized selection of a subset of the samples. This randomization turns the optimization problem into a stochastic one. We propose to consider the loss as a noisy observation with respect to some reference optimum. This interpretation of the loss allows us to adopt Kalman filtering as an optimizer, as its recursive formulation is designed to estimate unknown parameters from noisy measurements. Moreover, we show that the Kalman Filter dynamical model for the evolution of the unknown parameters can be used to capture the gradient dynamics of advanced methods such as Momentum and Adam. We call this stochastic optimization method KOALA, which is short for Kalman Optimization Algorithm with Loss Adaptivity. KOALA is an easy to implement, scalable, and efficient method to train neural networks. We provide convergence analysis and show experimentally that it yields parameter estimates that are on par with or better than existing state of the art optimization algorithms across several neural network architectures and machine learning tasks, such as computer vision and language modeling.

### Method

In machine learning, we are interested in minimizing the expected risk

<img src="https://latex.codecogs.com/svg.image?\min_{x&space;\in&space;R^n}&space;L(x),&space;\quad&space;\text{where}&space;L(x)&space;\doteq&space;E_{\xi&space;\sim&space;p(\xi)}&space;[l(\xi;&space;x)]," /> 

with respect to some loss ℓ that is a function of both the data
ξ ∈ R
d with d the data dimensionality, p(ξ) is the probability density function of ξ, and the model parameters x ∈ R^n
(e.g., the weights of a neural network), where n is the number of parameters in the model.

We model the uncertainty of the identified parameters <img src="https://latex.codecogs.com/svg.image?x_k" /> 
 
