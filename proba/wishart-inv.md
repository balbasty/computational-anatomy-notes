---
layout:  default
title:   Inverse-Wishart
mathjax: true
---

Inverse-Wishart
===============

Ususal parameterisation
-----------------------

### Probability distribution function

The Inverse-Wishart distribution of dimension $K$ is defined over $K \times K$ positive definite matrices. Its parameters are $\boldsymbol{\Psi}$, its scale matrix, and $\nu > K - 1$, its degrees of freedom. The mean exists only for $\nu > K + 1$.

$$\begin{split}
    \mathcal{W}^{-1}_K\left(\mathbf{S} \mid \boldsymbol{\Psi}, \nu\right)
    & = \frac{\left|\boldsymbol{\Psi}\right|^{\nu/2}\left|\mathbf{S}\right|^{-(\nu+K+1)/2} \exp\left(-\frac{1}{2}\mathrm{Tr}\left(\boldsymbol{\Psi}\mathbf{S}^{-1}\right)\right)}{2^{\nu K/2}\Gamma_K\left(\frac{\nu}{2}\right)}
    \\
    \ln\mathcal{W}^{-1}_K\left(\mathbf{S} \mid \boldsymbol{\Psi}, \nu\right)
    & = \frac{\nu}{2} \ln\det\boldsymbol{\Psi}
    -\frac{\nu+K+1}{2}\ln\det\mathbf{S}
    - \frac{1}{2}\mathrm{Tr}\left(\boldsymbol{\Psi}\mathbf{S}^{-1}\right) \\
    & \phantom{ {}={} }
    - \frac{\nu K}{2} \ln 2
    - \ln\Gamma_K\left(\frac{\nu}{2}\right)
    \\
    \mathbb{E}\left[\mathbf{S}\right]
    & = \frac{\boldsymbol{\Psi}}{\nu-K-1}
    \\
    \mathbb{E}\left[\mathbf{S}^{-1}\right]
    & = \frac{\boldsymbol{\Psi}^{-1}}{\nu}
    \\
    \mathbb{E}\left[\ln\det\mathbf{S}\right]
    & = -\left(\psi_K\left(\frac{\nu}{2}\right) + K\ln 2 + \ln\det\boldsymbol\Psi\right)
    \\
    \mathbb{V}\left[\ln\det\mathbf{S}\right]
    & = \sum_{i=1}^K \psi_1\left(\frac{\nu+1-i}{2}\right)
\end{split}$$

This distribution always has a mode:

$$\max_{\mathbf{S}} \mathcal{W}^{-1}_K\left(\mathbf{S} \mid \boldsymbol{\Psi}, \nu\right) = \frac{\boldsymbol{\Psi}}{\nu+K+1}$$

### Maximum likelihood estimators

Let $(\mathbf{S}_n)$ a set of observed realisations from a Gamma distribution.

| $\hat{\boldsymbol{\Psi}} \mid (\mathbf{S}_n), \nu$ | $= \nu\left[\overline{\mathbf{S}^{-1}}\right]^{-1}$ |
| $\hat{\nu} \mid (\mathbf{S}_n)$             | solution of: $K \ln \hat{\nu} - \psi_K\left(\frac{\hat{\nu}}{2}\right) = K \ln 2 + \ln\det\overline{\mathbf{S}^{-1}} + \overline{\ln \det \mathbf{S}}$ |
| $\hat{\boldsymbol{\Psi}} \mid (\mathbf{S}_n)$      | $= \hat{\boldsymbol{\Psi}} \mid (\mathbf{S}_n), \hat{\nu}$ |

where

$$\overline{\mathbf{S}} = \frac{1}{N}\sum_{n=1}^N \mathbf{S}_n$$

$$\overline{\mathbf{S}^{-1}} = \frac{1}{N}\sum_{n=1}^N \mathbf{S}_n^{-1}$$

$$\overline{\ln\det\mathbf{S}} = \frac{1}{N}\sum_{n=1}^N \ln\det\mathbf{S}_n$$

There is no closed form solution for $\hat{\nu}$, but an approximate solution can be found by numerical optimisation.

*I need to check my math for $\nu$*

### Conjugate prior

We list here the distributions that can be used as conjugate prior for the parameters of an univariate Normal distribution:

| $\boldsymbol{\Psi} \mid \nu$ | [Wishart]({{site.baseurl}}/proba/wishart) | $\mathcal{W}$ |

Update equations can be found in the [Conjugate prior]({{site.baseurl}}/conjugate-prior) article.

### Kullback-Leibler divergence

The KL-divergence can be written as

$$\begin{split}
    \mathrm{KL}\left(q \middle\| p\right)
    & = \mathbb{E}_q\left[\ln q(x)\right] - \mathbb{E}_q\left[\ln p(x)\right]
    \\
    & = H\left(q, p\right) - H\left(q\right)
\end{split}$$

where $H$ is the cross-entropy. We have

$$\begin{split}
    H\left(\boldsymbol{\Psi}_1, \nu_1 \middle\| \boldsymbol{\Psi}_0, \nu_0\right)
    & = -\frac{\nu_0}{2}\ln\det\boldsymbol{\Psi}_0
    - \frac{\nu_0+K+1}{2}\ln\det\boldsymbol{\Psi}_1
    + \frac{1}{2\nu_1}\mathrm{Tr}\left(\boldsymbol{\Psi}_0\boldsymbol{\Psi}_1^{-1}\right)
    \\
    & \phantom{ {}={} }
    - \frac{K(K+1)}{2}\ln 2
    + \ln\Gamma_K\left(\frac{\nu_0}{2}\right)
    - \frac{\nu_0 + K + 1}{2}\psi_K\left(\frac{\nu_1}{2}\right)
    \\
    H\left(\boldsymbol{\Psi}_1, \nu_1 \right)
    & = -\frac{2\nu_1+K+1}{2}\ln\det\boldsymbol{\Psi}_1
    + \frac{K}{2\nu_1}
    \\
    & \phantom{ {}={} }
    - \frac{K(K+1)}{2}\ln 2
    + \ln\Gamma_K\left(\frac{\nu_1}{2}\right)
    - \frac{\nu_1 + K + 1}{2}\psi_K\left(\frac{\nu_1}{2}\right)
\end{split}$$

$$\boxed{\begin{split}
    \mathrm{KL}\left(\boldsymbol{\Psi}_1, \nu_1 \middle\| \boldsymbol{\Psi}_0, \nu_0\right)
    ={} & -\frac{\nu_0}{2}\ln\det\boldsymbol{\Psi}_0 + \frac{2\nu_1 - \nu_0}{2}\ln\det\boldsymbol{\Psi}_1
    + \frac{1}{2\nu_1}\left(\mathrm{Tr}\left(\boldsymbol{\Psi}_0\boldsymbol{\Psi}_1^{-1}\right) - K\right)\\
    & + \frac{\nu_1 - \nu_0}{2}\psi_K\left(\frac{\nu_1}{2}\right)
    + \ln\Gamma_K\left(\frac{\nu_0}{2}\right)
    - \ln\Gamma_K\left(\frac{\nu_1}{2}\right)
\end{split}}$$

*Needs to be checked*

"Normal covariance matrix conjugate" parameterisation
-----------------------------------------------------

*I am not sure that this is the best prameterisation yet.*

Another parameterisation, which may feel more natural when using the Wishart distribution as a prior for the precision matrix of a multivariate Gaussian distribution, uses the expected matrix instead of the scale matrix. This parameterisation only makes sense if $\nu > K + 1$:

$$\begin{split}
    \mathcal{W}^{-1}_K\left(\mathbf{S} \mid \boldsymbol{\Psi}, \nu\right)
    & {}= \frac{(\nu-K-1)^{\nu K/2}\left|\boldsymbol{\Psi}\right|^{\nu/2}}{2^{\nu K/2}\Gamma_K\left(\frac{\nu}{2}\right)}\left|\mathbf{S}\right|^{-\frac{\nu+K+1}{2}} \exp\left(-\frac{\nu-K-1}{2}\mathrm{Tr}\left(\boldsymbol{\Psi}\mathbf{S}^{-1}\right)\right)
    \\
    \ln\mathcal{W}^{-1}_K\left(\mathbf{S} \mid \boldsymbol{\Psi}, \nu\right)
    & {}= \frac{\nu}{2} \ln\det\boldsymbol{\Psi}
    -\frac{\nu+K+1}{2}\ln\det\mathbf{S}
    - \frac{\nu-K-1}{2}\mathrm{Tr}\left(\boldsymbol{\Psi}\mathbf{S}^{-1}\right) \\
    & \phantom{ {}={} } + \frac{\nu K}{2} \left(\ln(\nu-K-1) - \ln 2\right)
    - \ln\Gamma_K\left(\frac{\nu}{2}\right)
    \\
    \mathbb{E}\left[\mathbf{S}\right]
    & = \boldsymbol\Sigma
    \\
    \mathbb{E}\left[\mathbf{S}^{-1}\right]
    & = \frac{\nu-K-1}{\nu}\boldsymbol{\Psi}^{-1}
    \\
    \mathbb{E}\left[\ln\det\mathbf{S}\right]
    & = -\left(\psi_K\left(\frac{\nu}{2}\right) - K\ln \frac{\nu}{2} + \ln\det\boldsymbol\Sigma\right)
    \\
    \mathbb{V}\left[\ln\det\mathbf{S}\right]
    & = \sum_{i=1}^K \psi_1\left(\frac{\nu+1-i}{2}\right)
\end{split}$$

This distribution always has a mode:

$$\max_{\mathbf{S}} \mathcal{W}^{-1}_K\left(\mathbf{S} \mid \boldsymbol\Sigma, \nu\right) = \frac{\nu-K-1}{\nu+K+1}\boldsymbol\Sigma$$

### Maximum likelihood estimators

Let $(\mathbf{S}_n)$ a set of observed realisations from a Gamma distribution.

| $\hat{\boldsymbol\Sigma} \mid (\mathbf{S}_n)$ | $= \frac{\nu}{\nu-K-1}\left[\overline{\mathbf{S}^{-1}}\right]^{-1}$ |
| $\hat{\nu} \mid (\mathbf{S}_n)$                | solution of: $K \ln \hat{\nu} - \psi_K\left(\frac{\hat{\nu}}{2}\right) = K \ln 2 + \ln\det\overline{\mathbf{S}} - \overline{\ln \det \mathbf{S}}$ |
| $\hat{\boldsymbol{\Sigma}} \mid (\mathbf{S}_n)$      | $= \hat{\boldsymbol{\Sigma}} \mid (\mathbf{S}_n), \hat{\nu}$ |

where

$$\overline{\mathbf{S}} = \frac{1}{N}\sum_{n=1}^N \mathbf{S}_n$$

$$\overline{\mathbf{S}^{-1}} = \frac{1}{N}\sum_{n=1}^N \mathbf{S}_n^{-1}$$

$$\overline{\ln\det\mathbf{S}} = \frac{1}{N}\sum_{n=1}^N \ln\det\mathbf{S}_n$$

There is no closed form solution for $\hat{\nu}$, but an approximate solution can be found by numerical optimisation.

*I need to check my math for $\nu$*

### Kullback-Leibler divergence

The KL divergence becomes

$$\boxed{\begin{split}
    \mathrm{KL}\left(\boldsymbol{\Sigma}_1, \nu_1 \middle\| \boldsymbol{\Sigma}_0, \nu_0\right)
    ={} & -\frac{\nu_0}{2}\ln\det\boldsymbol{\Psi}_0 + \frac{2\nu_1 - \nu_0}{2}\ln\det\boldsymbol{\Psi}_1
    + \frac{1}{2\nu_1}\left(\frac{\nu_0-K-1}{\nu_1-K-1}\mathrm{Tr}\left(\boldsymbol{\Psi}_0\boldsymbol{\Psi}_1^{-1}\right) - K\right)\\
    & -\frac{\nu_0}{2}K\ln(\nu_0-K-1) + \frac{2\nu_1 - \nu_0}{2}K\ln(\nu_1-K-1) \\
    & + \frac{\nu_1 - \nu_0}{2}\psi_K\left(\frac{\nu_1}{2}\right)
    + \ln\Gamma_K\left(\frac{\nu_0}{2}\right)
    - \ln\Gamma_K\left(\frac{\nu_1}{2}\right)
\end{split}}$$

*Needs to be checked*

***

*Created by Yaël Balbastre on 10 April 2018. Last edited on 10 April 2018.*

***
