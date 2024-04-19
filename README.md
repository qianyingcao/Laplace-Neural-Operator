# Laplace Neural Operator

This repo contains a PyTorch implementation for the paper [Laplace Neural Operator for Solving Differential Equations](https://arxiv.org/abs/2303.10528)
by [Qianying Cao](https://scholar.google.com/citations?user=OrdbclEAAAAJ&hl=en&oi=sra), [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en&oi=sra), and [George Em Karniadakis](https://scholar.google.com/citations?user=yZ0-ywkAAAAJ&hl=en&oi=sra)

---

Laplace neural operator (LNO), which incorporates the pole-residue relationship between input-output spaces, leads to better interpretability and generalization for certain classes of problems. LNO is capable of processing non-periodic signals and transient responses resulting from simultaneously zero and non-zero initial conditions, which makes it achieve better approximation accuracy over other neural operators for extrapolation circumstances in solving several ODEs and PDEs. Moreover, LNO has good interpolation ability from a low-resolution input to high-resolution outputs at arbitrary locations within the domain. To demonstrate the scalability of LNO, large-scale simulations of Rossby waves around the globe, employing millions of degrees of freedom are conducted. Taken together, a pre-trained LNO model offers an effective real-time solution for general ODEs and PDEs at scale and is an efficient alternative to existing NOs.

# Requirements
torch                1.13.1+cu117


# Code
This website offers two versions of LNO code. The first version is used in all examples except Burger equation. This code regards the system poles and residues as training parameters (described in the Methods section), which avoids the network overfitting and performs well in extrapolation problems.  Although writing system poles and residues into partial-fraction form avoids the network overfitting, the LNO cannot deeply learn due to this strict formulation in interpolation problem. The second version, which is used in Burger equation, remains system poles and residues as training parameters for the transient term (the first term), but the steady-state term follows the idea of FNO which regards FRF as the the training parameters (the second term). When implementing the proposed method, the transient and steady-state terms are decoupled, allowing more flexibility of the LNO method for operator learning.

# Data
The data for all examples except Burger equation, Brusselator and shallow-water equations are in the corresponding folders. Since the data for Burger equation, Brusselator and shallow-water equations are too big to upload, one can request the data by sending email to qianying_cao@brown.edu.


# Citations
```
@article{cao2023lno,
  title={Lno: Laplace neural operator for solving differential equations},
  author={Cao, Qianying and Goswami, Somdatta and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2303.10528},
  year={2023}
}
```
