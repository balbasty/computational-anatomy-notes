---
layout:  default
title:   Univariate Normal
mathjax: true
---

Univariate Normal
=================

Usual parameterisation
----------------------

### Probability distribution function

The parameters of an univariate Gaussian distribution are $\mu$, its mean, and $\sigma^2$, its variance.

$$\begin{align*}
    \mathcal{N}\left(x \mid \mu, \sigma^2\right)
    & = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\left(x - \mu\right)^2}{2\sigma^2}\right)
    \\
    \ln\mathcal{N}\left(x \mid \mu, \sigma^2\right)
    & = -\frac{1}{2}\ln(2\pi\sigma^2) -\frac{\left(x - \mu\right)^2}{2\sigma^2}
    \\
    \mathbb{E}\left[x\right]
    & = \mu
    \\
    \mathbb{V}\left[x\right]
    & = \sigma^2
\end{align*}$$

It can also be parameterised by its precision $\lambda = \frac{1}{\sigma^2}$.

### Maximum likelihood estimators

Let $(x_n)$ be a set of observed realisations from a Normal distribution.

| $\hat{\mu} \mid (x_n)$           | $= \overline{x} = \frac{1}{N}\sum_{n=1}^N x_n$ | $\sim \mathcal{N}\left(\mu, \frac{\sigma^2}{N}\right)$ |
| $\hat{\sigma}^2 \mid (x_n)$      | $= \overline{x^2} - \overline{x}^2 = \frac{1}{N}\sum_{n=1}^N (x_n - \overline{x})^2$ | $\sim \frac{\sigma^2}{N} \chi^2\left(N-1\right)$ |
| $\hat{\sigma}^2 \mid \mu, (x_n)$ | $= \overline{x^2} + \mu(\mu - 2\overline{x}) = \frac{1}{N}\sum_{n=1}^N (x_n - \mu)^2$ | |

*I need to check $\hat{\sigma}^2 \mid \mu, (x_n)$*

### Conjugate prior

We list here the distributions that can be used as conjugate prior for the parameters of an univariate Normal distribution:


| $\mu \mid \lambda$  | [Univariate Normal]({{site.baseurl}}/proba/normal-uni)              | $\mathcal{N}_\lambda$ |
| $\lambda \mid \mu$  | [Gamma]({{site.baseurl}}/proba/gamma)                               | $\mathcal{G}_\mathcal{N}$ |
| $\sigma^2 \mid \mu$ | [~~Inverse-Gamma~~]({{site.baseurl}}/proba/gamma-inv)               | $\mathrm{Inv-}\mathcal{G}_\mathcal{N}$ |
| $\mu, \lambda$      | [~~Normal-Gamma~~]({{site.baseurl}}/proba/normal-gamma)             | $\mathcal{N}\mathcal{G}$ |
| $\mu, \sigma^2$     | [~~Normal-Inverse-Gamma~~]({{site.baseurl}}/proba/normal-gamma-inv) | $\mathcal{N}\mathrm{Inv-}\mathcal{G}$ |

Update equations can be found in the [Conjugate prior]({{site.baseurl}}/conjugate-prior) article.


### Kullback-Leibler divergence

The KL-divergence can be written as

$$\begin{align*}
    \mathrm{KL}\left(q ~\middle\|~ p\right)
    & = \mathbb{E}_q\left[\ln q(x)\right] - \mathbb{E}_q\left[\ln p(x)\right]
    \\
    & = H\left(q, p\right) - H\left(q\right)
\end{align*}$$

where $H$ is the cross-entropy. We have

$$\begin{align*}
    H\left(\mu_1, \sigma_1^2 ~\middle\|~ \mu_0, \sigma_0^2\right)
    & = \frac{1}{2}\left(\ln 2\pi + \ln \sigma_0^2 + \frac{\sigma_1^2}{\sigma_0^2} + \frac{(\mu_1 - \mu_0)^2}{\sigma_0^2}\right)
    \\
    H\left(\mu_1, \sigma_1^2\right)
    & = \frac{1}{2}\left(\ln 2\pi + \ln \sigma_1^2 + 1 \right)
\end{align*}$$

Consequently

$$\boxed{\mathrm{KL}\left(\mu_1, \sigma_1^2 ~\middle\|~ \mu_0, \sigma_0^2\right)
= \frac{1}{2}\left(\frac{\sigma_1^2}{\sigma_0^2} - \ln\frac{\sigma_1^2}{\sigma_0^2} +  \frac{(\mu_1 - \mu_0)^2}{\sigma_0^2} - 1\right)}$$

Or, if a parameterisation based on the precision is used,

$$\boxed{\mathrm{KL}\left(\mu_1, \lambda_1 ~\middle\|~ \mu_0, \lambda_0\right)
= \frac{1}{2}\left(\frac{\lambda_0}{\lambda_1} - \ln\frac{\lambda_0}{\lambda_1} +  \lambda_0(\mu_1 - \mu_0)^2 - 1\right)}$$


"Normal mean conjugate" parameterisation
----------------------------------------

When the Normal distribution is used as a conjugate prior for the mean of another Normal distribution with known precision $\lambda$, it makes sense to parameterise it in terms of its expected value, $\mu_0$, and degrees of freedom, $n_0$:

$$\begin{align*}
    \mathcal{N}_\lambda\left(\mu \mid \mu_0, n_0\right)
    & = \mathcal{N}\left(\mu \mid \mu_0, (n_0\lambda)^{-1}\right) = \sqrt{\frac{n_0\lambda}{2\pi}} \exp\left(-\frac{n_0\lambda}{2}\left(\mu - \mu_0\right)^2\right)
    \\
    \mathbb{E}\left[\mu\right]
    & = \mu_0
    \\
    \mathbb{V}\left[\mu\right]
    & = \frac{1}{n_0\lambda}
\end{align*}$$

### Kullback-Leibler divergence

The KL-divergence can be written as

$$\boxed{\mathrm{KL}_\lambda\left(\mu_1, n_1 ~\middle\|~ \mu_0, n_0\right)
= \frac{1}{2}\left(\frac{n_0}{n_1} - \ln\frac{n_0}{n_1} +  n_0\lambda(\mu_1 - \mu_0)^2 - 1\right)}$$

***

*Created by Yaël Balbastre on 6 April 2018. Last edited on 6 April 2018.*

***
