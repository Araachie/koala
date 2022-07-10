<h1 align="center">
  <br>
	[AAAI 2022] KOALA: A Kalman Optimization Algorithm with Loss Adaptivity
  <br>
</h1>
  <p align="center">
    <a href="https://araachie.github.io">Aram Davtyan</a> •
    <a href="https://www.cvg.unibe.ch/people/sameni">Sepehr Sameni</a> •
    <a href="https://www.cvg.unibe.ch/people/cerkezi">Llukman Cerkezi</a> •
    <a href="https://gmeishvili.github.io/">Givi Meishvili</a> •
    <a href="https://www.cvg.unibe.ch/people/bielski">Adam Bielski</a> •
    <a href="https://www.cvg.unibe.ch/people/favaro">Paolo Favaro</a>
  </p>
<h4 align="center">Official repository of the paper</h4>

<h4 align="center">AAAI 2022</h4>

<h4 align="center"><a href="https://araachie.github.io/koala/">Project Website</a> • <a href="https://arxiv.org/abs/2107.03331">Paper</a>

#
> **Abstract:** *Optimization is often cast as a deterministic problem,
> where the solution is found through some iterative procedure such as 
> gradient descent. However, when training neural networks the loss 
> function changes over (iteration) time due to the randomized selection 
> of a subset of the samples. This randomization turns the optimization 
> problem into a stochastic one. We propose to consider the loss as a noisy
> observation with respect to some reference optimum. This interpretation 
> of the loss allows us to adopt Kalman filtering as an optimizer, as its 
> recursive formulation is designed to estimate unknown parameters from 
> noisy measurements. Moreover, we show that the Kalman Filter dynamical 
> model for the evolution of the unknown parameters can be used to capture 
> the gradient dynamics of advanced methods such as Momentum and Adam. We 
> call this stochastic optimization method KOALA, which is short for 
> Kalman Optimization Algorithm with Loss Adaptivity. KOALA is an easy to 
> implement, scalable, and efficient method to train neural networks. We 
> provide convergence analysis and show experimentally that it yields 
> parameter estimates that are on par with or better than existing state 
> of the art optimization algorithms across several neural network 
> architectures and machine learning tasks, such as computer vision and 
> language modeling.*

## Citation

For citation we recommend using the following bibref.

Davtyan, A., Sameni, S., Cerkezi, L., Meishvili, G., Bielski, A., & Favaro, P. (2022). KOALA: A Kalman Optimization Algorithm with Loss Adaptivity. Proceedings of the AAAI Conference on Artificial Intelligence, 36(6), 6471-6479. https://doi.org/10.1609/aaai.v36i6.20599

    @article{Davtyan_Sameni_Cerkezi_Meishvili_Bielski_Favaro_2022,
        title   = {KOALA: A Kalman Optimization Algorithm with Loss Adaptivity},
        volume  = {36},
        url     = {https://ojs.aaai.org/index.php/AAAI/article/view/20599},
        DOI     = {10.1609/aaai.v36i6.20599},
        number  = {6},
        journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
        author  = {Davtyan, Aram and Sameni, Sepehr and Cerkezi, Llukman and Meishvili, Givi and Bielski, Adam and Favaro, Paolo},
        year    = {2022},
        month   = {Jun.},
        pages   = {6471-6479} 
    }

## Prerequisites

For convenience, we provide an `environment.yml` file that can be used to install the required packages to a `conda` environment with the following command `conda env create -f environment.yml`.

## How to use it

The main classes are implemented in `koala.py` (see `VanillaKOALA` and `MomentumKOALA`). For convenience, both are inherited from a base `KOALABase` class, which in its turn is inherited from `torch.optim.Optimizer`.
Note that in contrast to standard optimizers, KOALA's step is split into two stages `predict` and `update`.
An example of usage can be found in `training_funcs.train`.

## Training

To reproduce results from the paper, use the `train.py` script from the repository.
Usage example:

```
python train.py --optim koala-m --data cifar100 --num-gpus 2
```

