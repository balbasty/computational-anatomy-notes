---
layout:  default
title:   Wishart
mathjax: true
---

Wishart
=======

Ususal parameterisation
-----------------------

### Probability distribution function

The Wishart distribution of dimension $K$ is defined over $K \times K$ positive definite matrices. Its parameters are $\mathbf{V}$, its scale matrix, and $\nu > K - 1$, its degrees of freedom.

$$\begin{align*}
    \mathcal{W}_K\left(\mathbf{A} \mid \mathbf{V}, \nu\right)
    & = \frac{\left|\mathbf{A}\right|^{(\nu-K-1)/2} \exp\left(-\frac{1}{2}\mathrm{Tr}\left(\mathbf{V}^{-1}\mathbf{A}\right)\right)}{2^{\nu K/2}\left|\mathbf{V}\right|^{\nu/2}\Gamma_K\left(\frac{\nu}{2}\right)}
    \\
    \ln\mathcal{W}_K\left(\mathbf{A} \mid \mathbf{V}, \nu\right)
    & = \frac{\nu-K-1}{2}\ln\det\mathbf{A}
    - \frac{1}{2}\mathrm{Tr}\left(\mathbf{V}^{-1}\mathbf{A}\right)
    - \frac{\nu K}{2} \ln 2
    - \frac{\nu}{2} \ln\det\mathbf{V}
    - \ln\Gamma_K\left(\frac{\nu}{2}\right)
    \\
    \mathbb{E}\left[\mathbf{A}\right]
    & = \nu\mathbf{V}
    \\
    \mathbb{E}\left[\mathbf{A}^{-1}\right]
    & = (\nu-K-1)\mathbf{V}^{-1}
    \\
    \mathbb{E}\left[\ln\det\mathbf{A}\right]
    & = \psi_K\left(\frac{\nu}{2}\right) + K\ln 2 + \ln\det\mathbf{V}
    \\
    \mathbb{V}\left[\ln\det\mathbf{A}\right]
    & = \sum_{i=1}^K \psi_1\left(\frac{\nu+1-i}{2}\right)
\end{align*}$$

This distribution has a mode only if $\nu \geqslant K + 1$:

$$\max_{\mathbf{A}} \mathcal{W}_K\left(\mathbf{A} \mid \mathbf{V}, \nu\right) = (\nu-K-1)\mathbf{V}$$

### Maximum likelihood estimators

Let $(\mathbf{A}_n)$ a set of observed realisations from a Gamma distribution.

| $\hat{\mathbf{V}} \mid (\mathbf{A}_n), \nu$ | $= \frac{1}{\nu}\overline{\mathbf{A}}$ |
| $\hat{\nu} \mid (\mathbf{A}_n)$             | solution of: $K \ln \hat{\nu} - \psi_K\left(\frac{\hat{\nu}}{2}\right) = K \ln 2 + \ln\det\overline{\mathbf{A}} - \overline{\ln \det \mathbf{A}}$ |
| $\hat{\mathbf{V}} \mid (\mathbf{A}_n)$      | $= \hat{\mathbf{V}} \mid (\mathbf{A}_n), \hat{\nu}$ |

where

$$\overline{\mathbf{A}} = \frac{1}{N}\sum_{n=1}^N \mathbf{A}_n$$

$$\overline{\ln\det\mathbf{A}} = \frac{1}{N}\sum_{n=1}^N \ln\det\mathbf{A}_n$$

There is no closed form solution for $\hat{\nu}$, but an approximate solution can be found by numerical optimisation.

*I need to check my math for $\nu$*

### Conjugate prior

We list here the distributions that can be used as conjugate prior for the parameters of an univariate Normal distribution:

| $\mathbf{V} \mid \nu$ | [Inverse-Wishart]({{site.baseurl}}/proba/wishart-inv) | $\mathcal{W}^{-1}$ |

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
    H\left(\mathbf{V}_1, \nu_1 \middle\| \mathbf{V}_0, \nu_0\right)
    & = \frac{\nu_0}{2}\ln\det\mathbf{V}_0
    - \frac{\nu_0-K-1}{2}\ln\det\mathbf{V}_1
    + \frac{\nu_1}{2}\mathrm{Tr}\left(\mathbf{V}_0^{-1}\mathbf{V}_1\right)
    \\
    & \phantom{ {}={} }
    + \frac{K(K+1)}{2}\ln 2
    + \ln\Gamma_K\left(\frac{\nu_0}{2}\right)
    - \frac{\nu_0 - K - 1}{2}\psi_K\left(\frac{\nu_1}{2}\right)
    \\
    H\left(\mathbf{V}_1, \nu_1 \right)
    & = \frac{K+1}{2}\ln\det\mathbf{V}_1
    + \frac{\nu_1 K}{2}
    \\
    & \phantom{ {}={} }
    + \frac{K(K+1)}{2}\ln 2
    + \ln\Gamma_K\left(\frac{\nu_1}{2}\right)
    - \frac{\nu_1 - K - 1}{2}\psi_K\left(\frac{\nu_1}{2}\right)
\end{align*}$$

$$\boxed{\begin{split}
    \mathrm{KL}\left(\mathbf{V}_1, \nu_1 \middle\| \mathbf{V}_0, \nu_0\right)
    ={} & \frac{\nu_0}{2}\left(\ln\det\mathbf{V}_0 -\ln\det\mathbf{V}_1\right)
    + \frac{\nu_1}{2}\left(\mathrm{Tr}\left(\mathbf{V}_0^{-1}\mathbf{V}_1\right) - K\right)\\
    & + \frac{\nu_1 - \nu_0}{2}\psi_K\left(\frac{\nu_1}{2}\right)
    + \ln\Gamma_K\left(\frac{\nu_0}{2}\right)
    - \ln\Gamma_K\left(\frac{\nu_1}{2}\right)
\end{split}}$$

### Related distributions

| **Specialisation** |
| Exponential | $\mathrm{Exp}(\lambda) = \mathcal{W}\left(\frac{1}{2\lambda}, 2\right)$ |
| Chi-squared | $\chi^2(\nu) = \mathcal{G}\left(1, \nu\right)$ |
| Gamma       | $\mathcal{G}(\alpha, \beta) = \mathcal{W}\left(\frac{1}{2\beta}, 2\alpha\right)$ |
| **Power** |
| Inverse-Wishart | $\mathbf{X} \sim \mathcal{W}(\mathbf{V}, \nu) \Rightarrow \mathbf{X}^{-1} \sim \mathcal{W}^{-1}\left(\mathbf{V}^{-1}, \nu\right)$ |

"Normal precision matrix conjugate" parameterisation
----------------------------------------------------

Another parameterisation, which may feel more natural when using the Wishart distribution as a prior for the precision matrix of a multivariate Gaussian distribution, uses the expected matrix instead of the scale matrix:

$$\begin{align*}
    \mathcal{W}_K\left(\mathbf{A} \mid \boldsymbol\Lambda, \nu\right)
    & = \frac{\nu^{\nu K/2}\left|\mathbf{A}\right|^{(\nu-K-1)/2} \exp\left(-\frac{\nu}{2}\mathrm{Tr}\left(\boldsymbol\Lambda^{-1}\mathbf{A}\right)\right)}{2^{nK/2}\left|\boldsymbol\Lambda\right|^{\nu/2}\Gamma_K\left(\frac{\nu}{2}\right)}
    \\
    \ln\mathcal{W}_K\left(\mathbf{A} \mid \boldsymbol\Lambda, \nu\right)
    & = \frac{\nu K}{2}\ln \frac{\nu}{2}
    + \frac{\nu-K-1}{2}\ln\det\mathbf{A}
    - \frac{\nu}{2}\mathrm{Tr}\left(\boldsymbol\Lambda^{-1}\mathbf{A}\right)
    - \frac{\nu}{2} \ln\det\boldsymbol\Lambda
    - \ln\Gamma_K\left(\frac{\nu}{2}\right)
    \\
    \mathbb{E}\left[\mathbf{A}\right]
    & = \boldsymbol\Lambda
    \\
    \mathbb{E}\left[\mathbf{A}^{-1}\right]
    & = \frac{\nu-K-1}{\nu}\boldsymbol\Lambda^{-1}
    \\
    \mathbb{E}\left[\ln\det\mathbf{A}\right]
    & = \psi_K\left(\frac{\nu}{2}\right) - K\ln \frac{\nu}{2} + \ln\det\boldsymbol\Lambda
    \\
    \mathbb{V}\left[\ln\det\mathbf{A}\right]
    & = \sum_{i=1}^K \psi_1\left(\frac{\nu+1-i}{2}\right)
\end{align*}$$

This distribution has a mode only if $\nu \geqslant K + 1$:

$$\max_{\mathbf{A}} \mathcal{W}_K\left(\mathbf{A} \mid \boldsymbol\Lambda, \nu\right) = \frac{\nu-K-1}{\nu}\boldsymbol\Lambda$$

### Maximum likelihood estimators

Let $(\mathbf{A}_n)$ a set of observed realisations from a Gamma distribution.

| $\hat{\boldsymbol\Lambda} \mid (\mathbf{A}_n)$ | $= \overline{\mathbf{A}}$ |
| $\hat{\nu} \mid (\mathbf{A}_n)$                | solution of: $K \ln \hat{\nu} - \psi_K\left(\frac{\hat{\nu}}{2}\right) = K \ln 2 + \ln\det\overline{\mathbf{A}} - \overline{\ln \det \mathbf{A}}$ |

where

$$\overline{\mathbf{A}} = \frac{1}{N}\sum_{n=1}^N \mathbf{A}_n$$

$$\overline{\ln\det\mathbf{A}} = \frac{1}{N}\sum_{n=1}^N \ln\det\mathbf{A}_n$$

There is no closed form solution for $\hat{\nu}$, but an approximate solution can be found by numerical optimisation.

*I need to check my math for $\nu$*

### Kullback-Leibler divergence

The KL divergence becomes

$$\boxed{\begin{split}
    \mathrm{KL}_\mathcal{W}\left(\boldsymbol\Lambda_1, \nu_1 \middle\| \boldsymbol\Lambda_0, \nu_0\right)
    ={} & \frac{\nu_0}{2}\left(\ln\det\boldsymbol\Lambda_0 -\ln\det\boldsymbol\Lambda_1\right)
    + \frac{\nu_0}{2}\left(\ln \nu_1 - \ln \nu_0\right) \\
    & + \frac{\nu_0}{2}\mathrm{Tr}\left(\boldsymbol\Lambda_0^{-1}\boldsymbol\Lambda_1\right)
    - \frac{\nu_1 K}{2} \\
    & + \frac{\nu_1 - \nu_0}{2}\psi_K\left(\frac{\nu_1}{2}\right)
    + \ln\Gamma_K\left(\frac{\nu_0}{2}\right)
    - \ln\Gamma_K\left(\frac{\nu_1}{2}\right)
\end{split}}$$

***

*Created by Yaël Balbastre on 10 April 2018. Last edited on 10 April 2018.*

***
