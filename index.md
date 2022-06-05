<table class="center">
  <tr>
    <td>
      <p align="center">
        <b style="font-size: 24px">Paper:</b><br>
        <a href="https://arxiv.org/abs/2107.03331" style="font-size: 24px; text-decoration: none">[Arxiv]</a>
      </p>
    </td>
    <td>
      <p align="center">
        <b style="font-size: 24px">Code:</b><br>
        <a href="https://github.com/Araachie/koala" style="font-size: 24px; text-decoration: none">[GitHub]</a>
      </p>
    </td>
  </tr>
</table>

### Abstract

Optimization is often cast as a deterministic problem, where the solution is found through some iterative procedure such as gradient descent. However, when training neural networks the loss function changes over (iteration) time due to the randomized selection of a subset of the samples. This randomization turns the optimization problem into a stochastic one. We propose to consider the loss as a noisy observation with respect to some reference optimum. This interpretation of the loss allows us to adopt Kalman filtering as an optimizer, as its recursive formulation is designed to estimate unknown parameters from noisy measurements. Moreover, we show that the Kalman Filter dynamical model for the evolution of the unknown parameters can be used to capture the gradient dynamics of advanced methods such as Momentum and Adam. We call this stochastic optimization method KOALA, which is short for Kalman Optimization Algorithm with Loss Adaptivity. KOALA is an easy to implement, scalable, and efficient method to train neural networks. We provide convergence analysis and show experimentally that it yields parameter estimates that are on par with or better than existing state of the art optimization algorithms across several neural network architectures and machine learning tasks, such as computer vision and language modeling.

### Method

In machine learning, given the dataset ![](https://latex.codecogs.com/svg.image?\xi_i,&space;i=1\dots&space;m) and the loss function ![](https://latex.codecogs.com/svg.image?l(\xi;&space;x)), we are interested in minimizing the empirical risk with respect to network parameters ![](https://latex.codecogs.com/svg.image?x)), i.e., we want to find a ![](https://latex.codecogs.com/svg.image?\hat&space;x) such that

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\hat&space;L(\hat&space;x)=\min_{x}\hat&space;L(x),\quad\text{where}\quad\hat&space;L=\frac{1}{m}\sum_{i=1}^m&space;l(\xi_i;&space;x).">
</p>
  
Due to large datasets, SGD-like algorithms use minibatch risks

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\hat&space;L_k(x)=\frac{1}{|S_k|}\sum_{i\in&space;S_k}l(\xi_i;&space;x).">
</p>
  
Because of the central limit theorem, the minibatch loss ![](https://latex.codecogs.com/svg.image?\hat&space;L_k) tends to be Gaussian with mean the empirical loss ![](https://latex.codecogs.com/svg.image?\hat&space;L).

<p align="center">
<img width="420" src="https://user-images.githubusercontent.com/32042066/172056228-288c8da2-ca40-4d3e-94ff-3bc5f4d9c041.png">
</p>
  
For a feasible ![](https://latex.codecogs.com/svg.image?\hat&space;L^{\text{target}}) we can directly aim for ![](https://latex.codecogs.com/svg.image?\hat&space;L^{\text{target}}=\hat&space;L(\hat&space;x)).

<p align="center">
<img width="280" alt="Снимок экрана 2022-06-05 в 16 47 05" src="https://user-images.githubusercontent.com/32042066/172056265-f9351642-d4e7-41b1-b11b-2652acfcccd9.png">
</p>

We define training as the task of finding ![](https://latex.codecogs.com/svg.image?x_k) given the noisy minibatch risks ![](https://latex.codecogs.com/svg.image?\hat&space;L_k):

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\hat&space;L_k(x_k)=\hat&space;L^{\text{target}}-v_k.">
</p>

Thus, we formulate the empirical risk minimization as a loss adaptivity task.

By modeling the state dynamics (i.e., the network parameters) via a dynamical
system we can use the Extended Kalman Filtering equations to identify the parameters, which resembles into an optimization framework that we call KOALA.

With different state dynamics we derive two algorithms: KOALA-V (Vanilla) and KOALA-M (Momentum).

<p align="center">
<img width="420" alt="Снимок экрана 2022-06-05 в 16 52 06" src="https://user-images.githubusercontent.com/32042066/172056519-bd0f35e0-e7a3-4408-b341-fbd4de3f43c2.png">
</p>
  
For more details, please, check the <a href="https://arxiv.org/abs/2107.03331">paper</a>.

### Results

We have tested our algorithm against SGD and Adam on CIFAR-10/100 and ImageNet32. The results are shown in the table below. For more quantitative results, please, refer to the <a href="https://arxiv.org/abs/2107.03331">full text</a>.

<p align="center">
<img width="420" alt="Снимок экрана 2022-06-05 в 16 55 17" src="https://user-images.githubusercontent.com/32042066/172056675-55715b81-bfda-4d1b-9d8e-b5a65b3d9110.png">
</p>
