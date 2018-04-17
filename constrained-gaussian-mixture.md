---
layout:  default
title:   Constrained Gaussian mixture
mathjax: true
---


Encouraging similar covariance matrices between classes
=======================================================

This page is in reference to the article on [Hierarchical Gaussian mixtures]({{site.baseurl}}/hierarchical-gaussian-mixture).

## Model

Let $n \in 1 \dots N$ index subjects, $i \in 1 \dots I$ index voxels and $k \in 1 \dots K$ index classes. Let us write raw images $\mathbf{x}_n$. Let us assume that (expected) class belonging of each voxel is known and stored in $\mathbf{z}_n$ (*i.e.*, a "segmentation" or "responsibility" image). This is the situation that we have after the "Expectation" step of an EM (or variational EM) algorithm.

The conditional prior probability for each voxel is:

$$p\left(\mathbf{x}_{i,n} \mid \boldsymbol\mu_{1 \dots K, n}, \boldsymbol\Lambda_{1 \dots K,n}, \mathbf{z}_{i,n}\right) = \prod_{k=1}^K \mathcal{N}\left(\mathbf{x}_{i,n} \mid \boldsymbol\mu_{k, n}, \boldsymbol\Lambda_{k,n}\right)^{z_{k,i,n}}$$

Each gaussian is assumed to stem from a Normal-Wishart distribution:

$$\begin{split}p(\boldsymbol\mu_{k,n}, \boldsymbol\Lambda_{k,n} \mid \boldsymbol\mu_{k,0}, \lambda_{k,0}, \mathbf{W}_{k,0}, \nu_{k,0})
={} & \mathcal{N}\left(\boldsymbol\mu_{k,n} \mid \boldsymbol\mu_{k,0}, (\lambda_{k,0}\boldsymbol\Lambda_{k,n})^{-1}\right) \\
& \times \mathcal{W}\left(\boldsymbol\Lambda_{k,n} \mid \mathbf{W}_{k,0}, \nu_{k,0}\right)\end{split}$$

All prior inverse scale matrices stem from a shared Wishart distribution:

$$p\left(\mathbf{W}_{k,0}^{-1} \mid \boldsymbol\Psi, m\right) = \mathcal{W}\left(\mathbf{W}_{k,0}^{-1} \mid \boldsymbol\Psi_0, m_0\right)$$

We make the mean field approximation that the posterior factorises over $(\boldsymbol\mu_{k,n}, \boldsymbol\Lambda_{k,n})$, $\mathbf{W}_{k,0}^{-1}$ and $\boldsymbol\Psi_0$. We'll be searching for mode estimates for all the other variables ($\boldsymbol\mu\_{k,0}$, $\lambda\_{k,0}$, $\nu\_{k,0}$, $m_0$).

## Posterior

### Gaussian parameters (x Subject x Class)

$$q^\star(\boldsymbol\mu_{k,n}, \boldsymbol\Lambda_{k,n}) = \mathcal{N}\mathcal{W}\left(\boldsymbol\mu_{k,n}, \boldsymbol\Lambda_{k,n} \mid \boldsymbol\mu_{k,n}^\star, \lambda_{k,n}^\star, \mathbf{W}_{k,n}^\star, \nu_{k,n}^\star\right)$$

with

$$\begin{split}
\lambda_{k,n}^\star & = \lambda_{k,0} + \left[\sum_{i=1}^I z_{k,i,n}\right] \\
\nu_{k,n}^\star & = \nu_{k,0} + \left[\sum_{i=1}^I z_{k,i,n}\right] \\
\boldsymbol\mu_{k,n}^\star & = \frac{1}{\lambda_{k,n}^\star} \left(\lambda_{k,0}\boldsymbol\mu_{k,0} + \left[\sum_{i=1}^I z_{i,k,n}\mathbf{x}_{i,n}\right]\right) \\
\left(\mathbf{W}_{k,n}^\star\right)^{-1} & = \mathbb{E}\left[\mathbf{W}_{k,0}^{-1}\right] + \left[\sum_{i=1}^I z_{i,k,n}\mathbf{x}_{i,n}\mathbf{x}_{i,n}^\mathrm{T}\right] + \lambda_0\boldsymbol\mu_{k,0}\boldsymbol\mu_{k,0}^{\mathrm{T}} - \lambda_{k,n}^\star\boldsymbol\mu_{k,n}^\star\boldsymbol\mu_{k,n}^{\star\mathrm{T}}
\end{split}$$

yielding

$$\begin{split}
\mathbb{E}^\star\left[\boldsymbol\mu_{k,n}\right] & = \boldsymbol\mu_{k,n}^\star \\
\mathbb{E}^\star\left[\boldsymbol\Lambda_{k,n}\right] & = \nu_{k,n}^\star\mathbf{W}_{k,n}^\star \overset{\Delta}{=} \boldsymbol\Lambda_{k,n}^\star \\
\mathbb{E}^\star\left[\boldsymbol\mu_{k,n}\boldsymbol\mu_{k,n}^{\mathrm{T}}\right] & = \boldsymbol\mu_{k,n}^\star\boldsymbol\mu_{k,n}^{\star\mathrm{T}} + \left(\lambda_{k,n}^\star\nu_{k,n}^\star\mathbf{W}_{k,n}^\star\right)^{-1}
\end{split}$$

### Wishart scale matrix (x Class)

$$q^\star(\mathbf{W}_{k,0}^{-1}) = \mathcal{W}\left(\mathbf{W}_{k,0}^{-1} \mid \boldsymbol\Psi_k^\star, m_k^\star\right)$$

with

$$\begin{split}
m_k^\star & = m_0 + N\nu_{k,0} \\
\left(\boldsymbol\Psi_k^\star\right)^ {-1} & = \boldsymbol\Psi_0^{-1} + \sum_{n=1}^N \mathbb{E}^\star\left[\boldsymbol\Lambda_{k,n}\right] \\
\end{split}$$

yielding

$$\mathbb{E}^\star\left[\mathbf{W}_{k,0}^{-1}\right] = m_k^\star\boldsymbol\Psi_k^\star \overset{\Delta}{=} \mathbf{W}_{k,0}^{-1 \star}$$

$$\mathbb{E}^\star\left[\ln\det\mathbf{W}_{k,0}\right] = -\left(\psi_M\left(\frac{m_k^\star}{2}\right) + M\ln 2 + \ln\det\boldsymbol\Psi_k^\star\right)$$

## Lower bound

The lower bound on the model evidence (or negative free energy) can be written as

$$\begin{align}\mathcal{L} ={}
& \sum_{n=1}^N \sum_{k=1}^K \sum_{i=1}^I z_{i,k,n} \mathbb{E}^\star\left[\ln\mathcal{N}\left(\mathbf{x}_{i,n} \mid \boldsymbol\mu_{k,n}, \boldsymbol\Lambda_{k,n}\right)\right] \\
& - \sum_{n=1}^N \sum_{k=1}^K \mathbb{E}^\star\left[\mathrm{KL}\left(\boldsymbol\mu_{k,n}^\star, \lambda_{k,n}^\star, \mathbf{W}_{k,n}^\star, \nu_{k,n}^\star ~\middle\|~ \boldsymbol\mu_{k,0}, \lambda_{k,0}, \mathbf{W}_{k,0}, \nu_{k,0} \right)\right] \\
& - \sum_{k=1}^K \mathrm{KL}\left(\boldsymbol\Psi_k^\star, m_k^\star ~\middle\|~ \boldsymbol\Psi_0, m_0\right)
\end{align}$$

with

$$\begin{split}\mathbb{E}^\star\left[\ln\mathcal{N}\left(\mathbf{x}_{i,n} \mid \boldsymbol\mu_{k,n}, \boldsymbol\Lambda_{k,n}\right)\right]
={} & -\frac{M}{2}\ln 2\pi + \frac{1}{2}\mathbb{E}^\star\left[\ln\det\boldsymbol\Lambda_{k,n}\right] - \frac{M}{2\lambda_{k,n}^\star} \\
& - \frac{\nu_{k,n}^\star}{2}(\mathbf{x}_{i,n} - \boldsymbol\mu_{k,n}^\star)^\mathrm{T}\mathbf{W}_{k,n}^\star(\mathbf{x}_{i,n} - \boldsymbol\mu_{k,n}^\star)\end{split}$$

$$\begin{split}
& \mathbb{E}^\star\left[\mathrm{KL}\left(\boldsymbol\mu_{k,n}^\star, \lambda_{k,n}^\star, \mathbf{W}_{k,n}^\star, \nu_{k,n}^\star ~\middle\|~ \boldsymbol\mu_{k,0}, \lambda_{k,0}, \mathbf{W}_{k,0}, \nu_{k,0} \right)\right] \\
& {}=
\frac{1}{2}\left(M\frac{\lambda_{k,0}}{\lambda_{k,n}^\star} - M\ln\frac{\lambda_{k,0}}{\lambda_{k,n}^\star} + \lambda_{k,0}\nu_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})^{\mathrm{T}}\mathbf{W}_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0}) - M\right) \\
& \phantom{ {}={} }
+ \frac{\nu_{k,0}}{2}\left(\mathbb{E}^\star\left[\ln\det\mathbf{W}_{k,0}\right] - \ln\det\mathbf{W}_{k,n}^\star\right) \\
& \phantom{ {}={} }
+ \frac{\nu_{k,n}^\star}{2}\left(\mathrm{Tr}\left(\left(m_k^\star\boldsymbol\Psi_k^\star\right)^{-1}\mathbf{W}_{k,n}^\star\right) - M\right)\\
& \phantom{ {}={} }
+ \frac{\nu_{k,n}^\star - \nu_{k,0}}{2}\psi_M\left(\frac{\nu_{k,n}^\star}{2}\right) \\
& \phantom{ {}={} }
+ \ln\Gamma_M\left(\frac{\nu_{k,0}}{2}\right)
- \ln\Gamma_M\left(\frac{\nu_{k,n}^\star}{2}\right)
\end{split}$$

$$\begin{split}
\mathrm{KL}\left(\boldsymbol\Psi_k^\star, m_k^\star ~\middle\|~ \boldsymbol\Psi_0, m_0\right) ={}
& \frac{m_0}{2}\left(\ln\det\boldsymbol{\Psi}_0 - \ln\det\boldsymbol{\Psi}_k^\star\right) \\
& + \frac{m_k^\star}{2}\left(\mathrm{Tr}\left(\boldsymbol{\Psi}_0^{-1}\boldsymbol{\Psi}_k^\star\right) - M\right) \\
& + \frac{m_k^\star - m_0}{2}\psi_M\left(\frac{m_k^\star}{2}\right) \\
& + \ln\Gamma_M\left(\frac{m_0}{2}\right)
- \ln\Gamma_M\left(\frac{m_k^\star}{2}\right)
\end{split}$$

## Hyper-parameters optimisation

### Normal mean degrees of freedom ($\lambda_{k,0}$)

Keeping only terms that depend on $\lambda_{k,0}$, we have:

$$\mathcal{E} =
\frac{1}{2} \sum_{n=1}^N M\frac{\lambda_{k,0}}{\lambda_{k,n}^\star}
- M\ln\frac{\lambda_{k,0}}{\lambda_{k,n}^\star}
+ \lambda_{k,0}\nu_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})^{\mathrm{T}}\mathbf{W}_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0}) $$

By differentiating with respect to $\lambda_{k,0}$, we find:

$$\frac{\partial\mathcal{E}}{\partial\lambda_{k,0}} = \frac{1}{2}\sum_{n=1}^N \frac{M}{\lambda_{k,n}^\star} - \frac{M}{\lambda_{k,0}} + \nu_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})^{\mathrm{T}}\mathbf{W}_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})$$

Solving for a null derivative yields

$$
\lambda_{k,0} = MN\left(\sum_{n=1}^N \frac{M}{\lambda_{k,n}^\star} + \nu_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})^{\mathrm{T}}\mathbf{W}_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})\right)^{-1}
$$

### Normal mean ($\boldsymbol\mu_{k,0}$)

Keeping only terms that depend on $\boldsymbol\mu_{k,0}$, and substituting the above result for $\lambda_{k,0}$, we have:

$$\mathcal{E} = \frac{MN \sum_{n=1}^N \nu_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})^{\mathrm{T}}\mathbf{W}_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})}{\sum_{n=1}^N \frac{M}{\lambda_{k,n}^\star} + \nu_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})^{\mathrm{T}}\mathbf{W}_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})}$$

Let us write $\mathcal{S} = \sum_{n=1}^N \nu_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})^{\mathrm{T}}\mathbf{W}\_{k,n}^\star(\boldsymbol\mu_{k,n}^\star - \boldsymbol\mu_{k,0})$ and $\mathcal{L} = \sum_{n=1}^N \frac{M}{\lambda_{k,n}^\star}$ so that $\mathcal{E} = MN\frac{\mathcal{S}}{\mathcal{L} + \mathcal{S}}$, with

$$\begin{split}
\mathcal{S} ={}
& \left[\sum_{n=1}^N \nu_{k,n}^\star\mathrm{Tr}\left(\left[\boldsymbol\mu_{k,n}^\star\boldsymbol\mu_{k,n}^{\star\mathrm{T}}\right]\mathbf{W}_{k,n}^\star\right)\right]
- 2\mathrm{Tr}\left(\left[\sum_{n=1}^N \nu_{k,n}^\star \mathbf{W}_{k,n}^\star \boldsymbol\mu_{k,n}^\star\right]\boldsymbol\mu_{k,0}^\mathrm{T}\right) \\
& + \mathrm{Tr}\left(\left[\sum_{n=1}^N \nu_{k,n}^\star\mathbf{W}_{k,n}^\star\right]\left[\boldsymbol\mu_{k,0}\boldsymbol\mu_{k,0}^{\mathrm{T}}\right]\right)
\end{split}$$

$$\Rightarrow \frac{\partial\mathcal{S}}{\partial \boldsymbol\mu_{k,0}} = 2\left[\sum_{n=1}^N \nu_{k,n}^\star\mathbf{W}_{k,n}^\star\right]\boldsymbol\mu_{k,0} - 2\left[\sum_{n=1}^N \nu_{k,n}^\star \mathbf{W}_{k,n}^\star \boldsymbol\mu_{k,n}^\star\right]$$

Differentiating the complete objective function yields

$$\frac{\partial\mathcal{E}}{\partial\boldsymbol\mu_{k,0}} = \frac{\mathcal{S}'(\mathcal{L}+\mathcal{S}) - \mathcal{S}\mathcal{S}'}{(\mathcal{L}+\mathcal{S})^2} = \mathcal{S}'\frac{\mathcal{L}}{(\mathcal{L}+\mathcal{S})^2}$$

Consequently

$$\frac{\partial\mathcal{E}}{\partial\boldsymbol\mu_{k,0}} = 0 \Rightarrow \boldsymbol\mu_{k,0} = \left[\sum_{n=1}^N \nu_{k,n}^\star\mathbf{W}_{k,n}^\star\right]^{-1}\left[\sum_{n=1}^N \nu_{k,n}^\star \mathbf{W}_{k,n}^\star \boldsymbol\mu_{k,n}^\star\right]$$

### Normal variance degrees of freedom ($\nu_{k,0}$)

Keeping only terms that depend on $\nu_{k,0}$, we have:

$$\begin{split}\mathcal{E} = \sum_{n=1}^N
& \frac{\nu_{k,0}}{2}\left(\mathbb{E}^\star\left[\ln\det\mathbf{W}_{k,0}\right] - \ln\det\mathbf{W}_{k,n}^\star\right) \\
& - \frac{\nu_{k,0}}{2}\psi_M\left(\frac{\nu_{k,n}^\star}{2}\right)
+ \ln\Gamma_M\left(\frac{\nu_{k,0}}{2}\right)\end{split}$$

with:

$$\mathbb{E}^\star\left[\ln\det\mathbf{W}_{k,0}\right] = -\left(\psi_M\left(\frac{m_k^\star}{2}\right) + M\ln 2 + \ln\det\boldsymbol\Psi_k^\star\right)$$

yielding:

$$\begin{split}\mathcal{E} =
N\ln\Gamma_M\left(\frac{\nu_{k,0}}{2}\right)
- \frac{\nu_{k,0}}{2}\Bigg(
    & N\psi_M\left(\frac{m_k^\star}{2}\right) + \sum_{n=1}^N \psi_M\left(\frac{\nu_{k,n}^\star}{2}\right) \\
    & + N\ln\det\boldsymbol\Psi_k^\star + \sum_{n=1}^N \ln\det\mathbf{W}_{k,n}^\star \\
    & + NM\ln2
\Bigg)
\end{split}$$

Differentiating yields:

$$\begin{split}
g = \frac{\partial\mathcal{E}}{\partial\nu_{k,0}} =
\frac{N}{2}\Bigg(
    & \psi_M\left(\frac{\nu_{k,0}}{2}\right)
    - \psi_M\left(\frac{m_k^\star}{2}\right) - \frac{1}{N}\sum_{n=1}^N \psi_M\left(\frac{\nu_{k,n}^\star}{2}\right) \\
    & -\ln\det\boldsymbol\Psi_k^\star - \frac{1}{N}\sum_{n=1}^N \ln\det\mathbf{W}_{k,n}^\star \\
    & - M\ln2
\Bigg)
\end{split}$$

There is no closed form solution, so we'll use the gradient and Hessian of the objective function, with a Gauss-Newton optimisation scheme, instead:

$$H = \frac{\partial^2\mathcal{E}}{\partial\nu_{k,0}^2} = \frac{N}{4}\psi_M^{(1)}\left(\frac{\nu_{k,0}}{2}\right)$$

$$\nu_{k,0}^{(i+1)} = \nu_{k,0}^{(i)} - \frac{g^{(i)}}{H^{(i)}}$$

### Hyper-Wishart inverse scale matrix ($\boldsymbol\Psi_0^{-1}$)

Keeping only terms that depend on $\boldsymbol\Psi_0$, we have:

$$\mathcal{E} = -\frac{Km_0}{2}\ln\det\boldsymbol\Psi_0^{-1} + \frac{1}{2}\mathrm{Tr}\left(\boldsymbol\Psi_0^{-1}\left[\sum_{k=1}^K m_k^\star\boldsymbol\Psi_k^\star\right]\right)$$

Differentiating yields:

$$\frac{\partial\mathcal{E}}{\partial\boldsymbol\Psi_0^{-1}} = -\frac{Km_0}{2}\boldsymbol\Psi_0 + \frac{1}{2}\left[\sum_{k=1}^K m_k^\star\boldsymbol\Psi_k^\star\right]$$

$$\frac{\partial\mathcal{E}}{\partial\boldsymbol\Psi_0^{-1}} = 0 \Rightarrow \boldsymbol\Psi_0 = \frac{1}{m_0}\left[\frac{1}{K}\sum_{k=1}^K m_k^\star\boldsymbol\Psi_k^\star\right]$$

### Hyper-Wishart degrees of freedom ($m_0$)

Keeping only terms that depend on $m_0$, we have:

$$\mathcal{E} = \frac{m_0}{2}\left(K\ln\det\boldsymbol\Psi_0 - \sum_{k=1}^K\left[\ln\det\boldsymbol\Psi_k^\star + \psi_M\left(\frac{m_k^\star}{2}\right)\right]\right)
+ K\ln\Gamma_M\left(\frac{m_0}{2}\right)$$

> **Note:** In order to find the global optimum at once, I first tried substituting $\boldsymbol\Psi_0$ for its optimum value (above) that depends on $m_0$. However, this breaks the objective function's convexity, resulting in an ill-conditionned optimisation problem. It is thus preferable to alternate between optimising $m_0$ by Gauss-Newton and updating $\boldsymbol\Psi_0$ based on this new value.

Once again, we will resolve to using numerical optimisation:

$$g =
\frac{K}{2}\Bigg(
    \ln\det\boldsymbol\Psi_0
    - \frac{1}{K}\left[\sum_{k=1}^K\ln\det\boldsymbol\Psi_k^\star\right]
    - \frac{1}{K}\left[\sum_{k=1}^K\psi_M\left(\frac{m_k^\star}{2}\right)\right]
    + \psi_M\left(\frac{m_0}{2}\right)
\Bigg)$$

$$H = \frac{K}{4}\psi_M^{(1)}\left(\frac{m_0}{2}\right)$$

***

*Created by Yaël Balbastre on 11 April 2018. Last edited on 13 April 2018.*

***
