---
layout:  default
title:   Multivariate Normal
mathjax: true
---

Multivariate Normal
===================

Ususal parameterisation
-----------------------

### Probability distribution function

The parameters of a multivariate Gaussian distribution of dimension $K$ are $\boldsymbol\mu$, its mean, and $\boldsymbol\Sigma$, its covariance matrix. It can also be paramererised by $\boldsymbol\Lambda = \boldsymbol\Sigma^{-1}$, its precision matrix.

$$\begin{align*}
    \mathcal{N}\left(\mathbf{x} \mid \boldsymbol\mu, \boldsymbol\Sigma\right)
    & = \frac{1}{\sqrt{(2\pi)^K\det \boldsymbol\Sigma}} \exp\left(-\frac{1}{2}\left(\mathbf{x} - \boldsymbol\mu\right)^{\mathrm{T}}\boldsymbol\Sigma^{-1}\left(\mathbf{x} - \boldsymbol\mu\right)\right)
    \\
    \ln\mathcal{N}\left(\mathbf{x} \mid \boldsymbol\mu, \boldsymbol\Sigma\right)
    & = -\frac{K}{2}\ln(2\pi) -\frac{1}{2} \ln\det\boldsymbol\Sigma -\frac{1}{2}\left(\mathbf{x} - \boldsymbol\mu\right)^{\mathrm{T}}\boldsymbol\Sigma^{-1}\left(\mathbf{x} - \boldsymbol\mu\right)
    \\
    \mathbb{E}\left[\mathbf{x}\right]
    & = \boldsymbol\mu
    \\
    \mathrm{cov}\left[\mathbf{x}\right]
    & = \boldsymbol\Sigma
\end{align*}$$

### Maximum likelihood estimators

Let $(\mathbf{x}_n)$ be a set of observed realisations from a multivariate Normal distribution.

| $\hat{\boldsymbol\mu} \mid (\mathbf{x}_n)$                    | $= \overline{\mathbf{x}}$ | $= \frac{1}{N}\sum_{n=1}^N \mathbf{x}_n$ |
| $\hat{\boldsymbol\Sigma} \mid (\mathbf{x}_n)$                 | $= \overline{\mathbf{x}\mathbf{x}^{\mathrm{T}}} - \overline{\mathbf{x}}\overline{\mathbf{x}}^{\mathrm{T}}$ | $= \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n - \overline{\mathbf{x}})(\mathbf{x}_n - \overline{\mathbf{x}})^{\mathrm{T}}$ |
| $\hat{\boldsymbol\Sigma} \mid \boldsymbol\mu, (\mathbf{x}_n)$ | $= \overline{\mathbf{x}\mathbf{x}^{\mathrm{T}}} + (\boldsymbol\mu - 2\overline{\mathbf{x}})\boldsymbol\mu^{\mathrm{T}}$ | $= \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol\mu)(\mathbf{x}_n - \boldsymbol\mu)^{\mathrm{T}}$ |


*I need to check $\hat{\boldsymbol\Sigma} \mid \boldsymbol\mu, (\mathbf{x}_n)$*

### Conjugate prior

We list here the distributions that can be used as conjugate prior for the parameters of a multivariate Normal distribution:


| $\boldsymbol\mu \mid \boldsymbol\Lambda$ | [Multivariate Normal]({{site.baseurl}}/proba/normal-multi)              | $\mathcal{N}_{\boldsymbol\Lambda}$     |
| $\boldsymbol\Lambda \mid \boldsymbol\mu$ | [Wishart]({{site.baseurl}}/proba/wishart)                               | $\mathcal{W}_\mathcal{N}$              |
| $\boldsymbol\Sigma \mid \boldsymbol\mu$  | [Inverse-Wishart]({{site.baseurl}}/proba/wishart-inverse)               | $\mathrm{Inv-}\mathcal{W}_\mathcal{N}$ |
| $\boldsymbol\mu, \boldsymbol\Lambda$     | [~~Normal-Wishart~~]({{site.baseurl}}/proba/normal-wishart)                 | $\mathcal{NW}$                         |
| $\boldsymbol\mu, \boldsymbol\Sigma$      | [~~Normal-Inverse-Wishart~~]({{site.baseurl}}/proba/normal-wishart-inverse) | $\mathcal{N}\mathrm{Inv-}\mathcal{W}$  |

Update equations can be found in the [Conjugate prior]({{site.baseurl}}/conjugate-prior) article.

### Kullback-Leibler divergence

The KL-divergence can be written as

$$\begin{align*}
    \mathrm{KL}\left(q \middle\| p\right)
    & = \mathbb{E}_q\left[\ln q(x)\right] - \mathbb{E}_q\left[\ln p(x)\right]
    \\
    & = H\left(q, p\right) - H\left(q\right)
\end{align*}$$

where $H$ is the cross-entropy. We have

$$\begin{align*}
    H\left(\boldsymbol\mu_1, \boldsymbol\Sigma_1 \middle\| \boldsymbol\mu_0, \boldsymbol\Sigma_0\right)
    & = \frac{1}{2}\left(K\ln 2\pi + \ln\det\boldsymbol\Sigma_0 + \mathrm{Tr}\left(\boldsymbol\Sigma_0^{-1}\boldsymbol\Sigma_1\right) + \left(\boldsymbol\mu_1 - \boldsymbol\mu_0\right)^{\mathrm{T}}\boldsymbol\Sigma_0^{-1}\left(\boldsymbol\mu_1 - \boldsymbol\mu_0\right)\right)
    \\
    H\left(\boldsymbol\mu_1, \boldsymbol\Sigma_1\right)
    & = \frac{1}{2}\left(K\ln 2\pi + \ln\det\boldsymbol\Sigma_1 + K \right)
\end{align*}$$

Consequently

$$\boxed{\mathrm{KL}\left(\boldsymbol\mu_1, \boldsymbol\Sigma_1 \middle\| \boldsymbol\mu_0, \boldsymbol\Sigma_0\right)
= \frac{1}{2}\left(\mathrm{Tr}\left(\boldsymbol\Sigma_0^{-1}\boldsymbol\Sigma_1\right) - \ln\frac{\det\boldsymbol\Sigma_1}{\det\boldsymbol\Sigma_0} +  \left(\boldsymbol\mu_1 - \boldsymbol\mu_0\right)^{\mathrm{T}}\boldsymbol\Sigma_0^{-1}\left(\boldsymbol\mu_1 - \boldsymbol\mu_0\right) - K\right)}$$

Or, if a parameterisation based on the precision matrix is used,

$$\boxed{\mathrm{KL}\left(\boldsymbol\mu_1, \boldsymbol\Lambda_1 \middle\| \boldsymbol\mu_0, \boldsymbol\Lambda_0\right)
= \frac{1}{2}\left(\mathrm{Tr}\left(\boldsymbol\Lambda_1^{-1}\boldsymbol\Lambda_0\right) - \ln\frac{\det\boldsymbol\Lambda_0}{\det\boldsymbol\Lambda_1} +  \left(\boldsymbol\mu_1 - \boldsymbol\mu_0\right)^{\mathrm{T}}\boldsymbol\Lambda_0\left(\boldsymbol\mu_1 - \boldsymbol\mu_0\right) - K\right)}$$


"Normal mean conjugate" parameterisation
----------------------------------------

When the Normal distribution is used as a conjugate prior for the mean of another Normal distribution with known precision matrix $\boldsymbol{\Lambda}$, it makes sense to parameterise it in terms of its expected value, $\boldsymbol{\mu}_0$, and degrees of freedom, $n_0$:

$$\begin{align*}
    \mathcal{N}_{\boldsymbol{\Lambda}}\left(\boldsymbol{\mu} \mid \boldsymbol{\mu}_0, n_0\right)
    & = \mathcal{N}\left(\boldsymbol{\mu} \mid \boldsymbol{\mu}_0, (n_0\boldsymbol{\Lambda})^{-1}\right) = \sqrt{\left(\frac{n_0}{2\pi}\right)^K \det\boldsymbol\Lambda} \exp\left(-\frac{n_0}{2}\left(\boldsymbol\mu - \boldsymbol\mu_0\right)^{\mathrm{T}}\boldsymbol{\Lambda}\left(\boldsymbol\mu - \boldsymbol\mu_0\right)\right)
    \\
    \mathbb{E}\left[\mu\right]
    & = \boldsymbol\mu_0
    \\
    \mathbb{V}\left[\mu\right]
    & = \left(n_0\boldsymbol\Lambda\right)^{-1}
\end{align*}$$

### Kullback-Leibler divergence

The KL-divergence can be written as

$$\boxed{\mathrm{KL}_{\boldsymbol\Lambda}\left(\mu_1, n_1 ~\middle\|~ \mu_0, n_0\right)
= \frac{1}{2}\left(K\frac{n_0}{n_1} - K\ln\frac{n_0}{n_1} +  n_0(\boldsymbol\mu_1 - \boldsymbol\mu_0)^{\mathrm{T}}\boldsymbol\Lambda(\boldsymbol\mu_1 - \boldsymbol\mu_0) - K\right)}$$

***

*Created by Yaël Balbastre on 6 April 2018. Last edited on 9 April 2018.*

***
