---
layout:  page
title:   Variational inference
mathjax: true
---

Theory
------

In variational inference, we do not make any distinction between random parameters and latent variables. There are only latent variables, and we write their set $\mathbf{Z} = \left( \mathbf{Z}_i ; 1 \leqslant i \leqslant N \right)$. Let $\mathbf{X}$ be some observed variable of the model, supposing that we know the complete likelihood $p(\mathbf{X}, \mathbf{Z})$, the problem consists in finding a probability distribution $q(\mathbf{Z})$ that approximates $p(\mathbf{Z} \mid \mathbf{X})$.

Recall that making use of Bayes' rule,

$$p( \mathbf{Z} \mid \mathbf{X} ) = \frac{p( \mathbf{Z}, \mathbf{X} )}{p(\mathbf{X})} ~.$$

Since we known $p( \mathbf{Z}, \mathbf{X} )$, explicating $p(\mathbf{X})$ means explicating $p( \mathbf{Z} \mid \mathbf{X} )$ and *vice versa*.

Note that for *any* probability distribution $q(\mathbf{Z})$,

$$\begin{align*}
    \ln p( \mathbf{X} )
    & = \ln p( \mathbf{X} ) \int q( \mathbf{Z} ) \mathrm{d}\mathbf{Z}
    \\
    & = \int q( \mathbf{Z} ) \ln p( \mathbf{X} ) \mathrm{d}\mathbf{Z}
    \\
    & = \int q( \mathbf{Z} ) \Big( \ln p( \mathbf{X}, \mathbf{Z}) - \ln p( \mathbf{X}, \mathbf{Z}) +  \ln p( \mathbf{X} )  \Big)\mathrm{d}\mathbf{Z}
    \\
    & = \int q( \mathbf{Z} ) \Big( \ln p( \mathbf{X}, \mathbf{Z}) - \ln p( \mathbf{Z} \mid \mathbf{X} ) \Big)\mathrm{d}\mathbf{Z}
    \\
    & = \int q( \mathbf{Z} ) \Big( \ln p( \mathbf{X}, \mathbf{Z}) - \ln p( \mathbf{Z} \mid \mathbf{X} )
    + \ln q( \mathbf{Z} ) - \ln q( \mathbf{Z} ) \Big)\mathrm{d}\mathbf{Z}
    \\
    & = \int q( \mathbf{Z} ) \Big( \ln p( \mathbf{X}, \mathbf{Z}) - \ln p( \mathbf{Z} \mid \mathbf{X} )
    + \ln q( \mathbf{Z} ) - \ln q( \mathbf{Z} ) \Big)\mathrm{d}\mathbf{Z}
\end{align*}$$

We can split the above integral (convergence of the two sub-integrals will not be shown here):
$$\begin{align*}
    \ln p( \mathbf{X} )
    & = \int q( \mathbf{Z} ) \Big( \ln p( \mathbf{X}, \mathbf{Z}) - \ln q( \mathbf{Z} ) \Big)\mathrm{d}\mathbf{Z}
    & -
    & \int q( \mathbf{Z} ) \Big( \ln p( \mathbf{Z} \mid \mathbf{X} )
    - \ln q( \mathbf{Z} ) \Big)\mathrm{d}\mathbf{Z}
    \\
    & = \int q( \mathbf{Z} ) \ln\left\{\frac{p( \mathbf{X}, \mathbf{Z})}{q( \mathbf{Z} )} \right\}\mathrm{d}\mathbf{Z}
    & -
    & \int q( \mathbf{Z} ) \ln\left\{\frac{p( \mathbf{Z} \mid \mathbf{X} )}{q( \mathbf{Z} )} \right\} \mathrm{d}\mathbf{Z}
    \\
    & = \int q( \mathbf{Z} ) \ln\left\{\frac{p( \mathbf{X}, \mathbf{Z})}{q( \mathbf{Z} )} \right\}\mathrm{d}\mathbf{Z}
    & +
    & \int q( \mathbf{Z} ) \ln\left\{\frac{q( \mathbf{Z} )}{p( \mathbf{Z} \mid \mathbf{X} )} \right\} \mathrm{d}\mathbf{Z}
    \\
    & = \mathcal{L}(q)
    & +
    & \mathrm{KL}\left( q \middle\| p \right)
\end{align*}$$

Note that (1) the KL-divergence is always positive and (2) when $q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X})$, the KL-divergence is null and thus $\mathcal{L}(q) = \ln p(\mathbf{X})$. Consequently, $\mathcal{L}(q)$ defines a lower bound on the model log-evidence $\ln p(\mathbf{X})$, which "touches" the log-evidence when $q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X})$.

What is the point of searching for $q$ and not for the true posterior? Because it allows us to restrict ourselves to tractable posteriors. In particular, it is often practical to cluster latent variables into factor groups w.r.t. the approximate posterior, *i.e.* looking for $q$ of the form


$$q(\mathbf{Z}) = \prod_{i=1}^M q_i(\mathbf{Z}_i)$$

This is called a *mean field* approximation and means that for any $j \in [1, M]$ we can write

$$\begin{align*}
    \mathcal{L}(q)
    & = \int \prod_i q_i(\mathbf{Z}_i) \Big(
        \ln p(\mathbf{X}, \mathbf{Z})
        - \sum_k \ln q_k(\mathbf{Z}_k)
    \Big) \mathrm{d}\mathbf{Z} \\
    & = \int q_j(\mathbf{Z}_j) \prod_{i\neq j}q_i(\mathbf{Z}_i)
        \ln p(\mathbf{X}, \mathbf{Z}) \mathrm{d}\mathbf{Z} \\
    & \phantom{ {}={} }
        - \int q_j(\mathbf{Z}_j) \prod_{i\neq j}q_i(\mathbf{Z}_i)
        \Big(\sum_k \ln q_k( \mathbf{Z}_k) \Big) \mathrm{d} \mathbf{Z} \\
    & = \int q_j(\mathbf{Z}_j) \left( \int \ln p(\mathbf{X}, \mathbf{Z}) \prod_{i\neq j} q_i
         \mathrm{d}\mathbf{Z}_i \right) \mathrm{d}\mathbf{Z}_j \\
    & \phantom{ {}={} }
        -  \int q_j(\mathbf{Z}_j) \ln q_j(\mathbf{Z}_j) \prod_{i\neq j}q_i(\mathbf{Z}_i) \mathrm{d}\mathbf{Z} \\
    & \phantom{ {}={} }
        - \int q_j(\mathbf{Z}_j) \prod_{i\neq j}q_i(\mathbf{Z}_i) \Big(\sum_{k\neq j} \ln q_k( \mathbf{Z}_k) \Big) \mathrm{d} \mathbf{Z} \\
    & = \int q_j(\mathbf{Z}_j) \left( \int \ln p(\mathbf{X}, \mathbf{Z}) \prod_{i\neq j} q_i
         \mathrm{d}\mathbf{Z}_i \right) \mathrm{d}\mathbf{Z}_j \\
    & \phantom{ {}={} }
        - \int q_j(\mathbf{Z}_j) \ln q_j(\mathbf{Z}_j) \mathrm{d}\mathbf{Z}_j
        \prod_{i\neq j} \int q_i(\mathbf{Z}_i) \mathrm{d}\mathbf{Z}_i \\
    & \phantom{ {}={} }
        - \int q_j(\mathbf{Z}_j) \mathrm{d}\mathbf{Z}_j
        \int \prod_{i\neq j}q_i(\mathbf{Z}_i) \Big(\sum_{k\neq j} \ln q_k( \mathbf{Z}_k) \Big) \mathrm{d} \mathbf{Z}_i
\end{align*}$$

And since $\forall i, \int q_i(\mathbf{Z}_i) \mathrm{d}\mathbf{Z}_i = 1$,

$$\begin{align*}
    \mathcal{L}(q)
    & =
    \int q_j(\mathbf{Z}_j) \left( \int \ln p(\mathbf{X}, \mathbf{Z}) \prod_{i\neq j} q_i
         \mathrm{d}\mathbf{Z}_i \right) \mathrm{d}\mathbf{Z}_j
    -
    \int q_j(\mathbf{Z}_j) \ln q_j(\mathbf{Z}_j) \mathrm{d}\mathbf{Z}_j
    +
    \mathcal{C}_{q_j}
    \\
    & =
    \int q_j(\mathbf{Z}_j) \left( \int \ln p(\mathbf{X}, \mathbf{Z}) \prod_{i\neq j} q_i
         \mathrm{d}\mathbf{Z}_i - \ln q_j(\mathbf{Z}_j) \right) \mathrm{d}\mathbf{Z}_j
    +
    \mathcal{C}_{q_j}
    \\
    & =
    \int q_j(\mathbf{Z}_j) \Bigg( \ln \tilde{p}_j(\mathbf{X}, \mathbf{Z}_j)
         - \ln q_j(\mathbf{Z}_j) \Bigg) \mathrm{d}\mathbf{Z}_j
    +
    \mathcal{C}_{q_j}
    \\
    & = -\mathrm{KL}\left( q_j \middle\| \tilde{p}_j \right)
\end{align*}$$

where

$$\begin{align*}
    \ln \tilde{p}_j(\mathbf{X}, \mathbf{Z}_j)
    & = \int \ln p(\mathbf{X}, \mathbf{Z}) \prod_{i\neq j} q_i(\mathbf{Z}_i) \mathrm{d}\mathbf{Z}_i + \mathcal{C}_{\mathbf{Z}_j}
    \\
    & = \tilde{\mathbb{E}}_{\mathbf{Z}_{i \neq j}} \Big[ \ln p(\mathbf{X}, \mathbf{Z}) \Big] + \mathcal{C}_{\mathbf{Z}_j}
\end{align*}$$

For now on, $\tilde{\mathbb{E}}$ will denote an expected value w.r.t. approximate posteriors.

Keeping $q_{i\neq j}$ fixed, this quantity is thus minimised for $q^\star_j = \tilde{p}_j$, yielding:

$$\boxed{\ln q^\star_j(\mathbf{Z}_j) = \tilde{\mathbb{E}}_{\mathbf{Z}_{i \neq j}} \Big[ \ln p(\mathbf{X}, \mathbf{Z}) \Big] + \mathcal{C}_{\mathbf{Z}_j}}$$

Consequently, all approximate log-posteriors $q_i$ can be optimised in turn by keeping the others fixed and computing the above expected model log-likelihood. This insures that the lower bound improved after every step, and *"convergence is guaranteed because bound is convex with respect to each of the factors"* according to [Bishop (2006)](https://www.springer.com/us/book/9780387310732), who references [Boyd and Vandenberghe (2004)](http://web.stanford.edu/~boyd/cvxbook/).

However, this expected value might be tricky to calculate. Ther are three general cases:

- If a posterior $q_j$ is well parameterised thanks to the use of a conjugate prior, it insures that the posterior possesses the same form and is thus entirely characterised by a few parameters. For example, if, in the generative model, one conditional probability $p(\mathbf{Z}_i \mid \mathbf{Z}_j)$ is a centred multivariate normal of precision matrix $\mathbf{Z}_j$, then choosing a Wishart distribution for $p(\mathbf{Z}_j)$ insures that $q_j(\mathbf{Z}_j)$ is also a Wishart distribution.

- Sometimes, we are not interested in the complete posterior of a variable, but only in its expected value (or higher order moments that are involved when computing the approximate posterior of the other variables according to the update scheme presented above). In this case, when the posterior has no known form, Monte Carlo sampling can be used to approximate these moments.

- Finally, if we are mainly interested in the maximum value (or *mode*) of a posterior distribution, a Laplace approximation can be made, meaning that we suppose that $q_j(\mathbf{Z}_j) \approx \mathcal{N}\left(\mathbf{Z}_j \mid \boldsymbol{\mu}, \mathbf{A}^{-1}\right)$, where $\boldsymbol{\mu}$ is the maximum of the log-likelihood $\ln q_j$ and $\mathbf{A}$ is the Hessian of $\ln q_j$ w.r.t. $\mathbf{Z}_j$ about that maximum. Both values can then be found by numerical optimisation. This is very useful when the expectation of the model likelihood is calculable and differentiable at any point, but possesses a non-standard form. The use of Laplace approximation in variational frameworks is sometimes called variational Laplace (VL) in the literature.

Note that using VL violates the assumptions of the variational framework and might thus lead to a biased estimation. This is something to keep in mind. In this VL framework, it it possible to find an approximate mean that only improves (instead of maximising) the posterior, as in the Generalised-EM (GEM) framework.

Lower bound
-----------

Another writing of the lower bound is

$$\begin{align*}
    \mathcal{L}(q)
    & =
    \int q(\mathbf{Z}) \ln p(\mathbf{X}, \mathbf{Z}) \mathrm{d}\mathbf{Z}
    -
    \int q(\mathbf{Z}) \ln q(\mathbf{Z}) \mathrm{d}\mathbf{Z}
    \\
    & =
    \int \ln p(\mathbf{X}, \mathbf{Z}) \prod_i q(\mathbf{Z}_i)  \mathrm{d}\mathbf{Z}_i
    -
    \int \prod_i q(\mathbf{Z}_i) \left(\sum_j \ln q(\mathbf{Z}_j)\right) \mathrm{d}\mathbf{Z}_i
    \\
    & =
    \tilde{\mathbb{E}}_{\mathbf{Z}}\Big[ \ln p(\mathbf{X}, \mathbf{Z})\Big]
    -
    \sum_j \int \ln q_j(\mathbf{Z}_j) \prod_i q_i(\mathbf{Z}_i) \mathrm{d}\mathbf{Z}_i
    \\
    & =
    \tilde{\mathbb{E}}_{\mathbf{Z}}\Big[ \ln p(\mathbf{X}, \mathbf{Z})\Big]
    -
    \sum_j \int q_j(\mathbf{Z}_j) \ln q_j(\mathbf{Z}_j) d\mathbf{Z}_j \prod_{i\neq j} \int q_i(\mathbf{Z}_i) \mathrm{d}\mathbf{Z}_i
\end{align*}$$

$$\boxed{\mathcal{L}(q) = \tilde{\mathbb{E}}_{\mathbf{Z}}\Big[ \ln p(\mathbf{X}, \mathbf{Z})\Big] - \sum_j \tilde{\mathbb{E}}_{\mathbf{Z}_j} \Big[ \ln q_j(\mathbf{Z}_j) \Big]}$$

This form is often simpler to calculate and allows tracking the lower bound to check its growth and convergence. If all the $q_i$ are parameterised, it may also be simpler to maximise this form of the lower bound in turn with respect to the parameters of each factor.

In practice, if we fully decompose the model likelihood in terms of known prior conditional probabilities, that we will write $p_j$, the above lower bound can be written

$$\boxed{\mathcal{L}(q) = \tilde{\mathbb{E}}_\mathbf{Z}\Big[\ln p(\mathbf{X} \mid \mathbf{Z})\Big] - \sum_j \tilde{\mathbb{E}}_{\setminus\mathbf{Z}_j}\Big[\mathrm{KL}\left( q_j |\middle| p_j \right)\Big]}$$

which can be understood as the data likelihood minus the divergence of the posteriors from the (mean) conditional priors. When a conjugate prior is used so that $p_j$ and $q_j$ have the same form, the KL-divergence is also known. Formulas for common distributions are provided in the [Conjugate prior]({{site.baseurl}}/conjugate-prior) article.

Notations
---------

Let me introduce a few notations regarding variational inference that I will use in subsequent sections. This will allow me to not repeat myself later on, and will hopefully help the reader getting more familiar with the equations.

Notations using $i$ and $j$ are useful when describing the general variational theory, but becomes less practical when it comes to real models, were we like to use different letters to designate different variables. In general, I will give the assumed factorisation along with the graphical model and the model likelihood.

- For a given latent variable $\mathbf{Y}$, which is a factor of its own in the mean field approximations, when no ambiguity exist we will assume

   $$q_\mathbf{Y}(\mathbf{Y}) = q(\mathbf{Y}) = q_\mathbf{Y} \approx p( \mathbf{Y} \mid \mathbf{X} )$$

- The current best estimate for a posterior $q_\mathbf{Y}$ will be denoted $q^\star_\mathbf{Y}$.

- If $\mathbf{Z}$ denotes the set of all latent variables and $\mathbf{Y}$ is one of the mean field factor, then

   $$\tilde{\mathbb{E}}_{\setminus\mathbf{Y}}\left[ \mathbf{f}(\mathbf{Z}) \right]
= \tilde{\mathbb{E}}_{\mathbf{Z}\setminus\mathbf{Y}}\left[ \mathbf{f}(\mathbf{Z}) \right]
= \int \mathbf{f}(\mathbf{Z}) \prod_{\mathbf{Z} \in \mathbf{Z} \setminus \mathbf{Y}} q(\mathbf{Z}) \mathrm{d}\mathbf{Z}$$

- When the best estimates of $q$ are used in the integral, we will write

   $$\mathbb{E}^\star_{\setminus\mathbf{Y}}\left[ \mathbf{f}(\mathbf{Z}) \right]
= \mathbb{E}^\star_{\mathbf{Z}\setminus\mathbf{Y}}\left[ \mathbf{f}(\mathbf{Z}) \right]
= \int \mathbf{f}(\mathbf{Z}) \prod_{\mathbf{Z} \in \mathbf{Z} \setminus \mathbf{Y}} q^\star(\mathbf{Z}) \mathrm{d}\mathbf{Z}$$

- Due to the mean field approximation, for any factor $\mathbf{Y}$ and any function $\mathbf{f}$ that do not depend on any factor other that $\mathbf{Y}$,

   $$\tilde{\mathbb{E}}_{\mathbf{Z}}\left[ \mathbf{f}(\mathbf{Y}) \right]
= \tilde{\mathbb{E}}_{\mathbf{Y}}\left[ \mathbf{f}(\mathbf{Y}) \right]
= \int \mathbf{f}(\mathbf{Y}) q(\mathbf{Y}) \mathrm{d}\mathbf{Y}$$

   In this case, when there is no ambiguity, we will write

$$\tilde{\mathbb{E}}_{\mathbf{Y}}\left[ \mathbf{f}(\mathbf{Y}) \right] = \tilde{\mathbb{E}}\left[ \mathbf{f}(\mathbf{Y}) \right]$$

-  Similarly, if $\mathbf{Y}$ and $\mathbf{Z}$ are two factors and $\mathbf{f}$ is a function such that $$\mathbf{f}(\mathbf{Y}, \mathbf{Z}) = \mathbf{f}_\mathbf{Y} (\mathbf{Y}) ~ \mathbf{f}_\mathbf{Z} (\mathbf{Z})$$, then

   $$\tilde{\mathbb{E}}_{\mathbf{Y}, \mathbf{Z}}\left[ \mathbf{f}(\mathbf{Y}, \mathbf{Z}) \right]
   =\tilde{\mathbb{E}}_{\mathbf{Y}}\left[\mathbf{f}_\mathbf{Y}(\mathbf{Y})\right]
   \tilde{\mathbb{E}}_{\mathbf{Z}}\left[\mathbf{f}_\mathbf{Z}(\mathbf{Z})\right]$$

The use of $\tilde{\mathbb{E}}$ or $\mathbb{E}^\star$ allows us to easily make the difference between expected values with respect to the posterior or with respect to the prior (in which case we will simply use $\mathbb{E}$).

Mode estimate
-------------

When dealing with high dimensional variables (such as images), it is sometimes easier to look for mode estimate. A "reasonably Bayesian" approach will use variational Bayes when possible, and mode (*i.e.* kind of MAP) estimates when a Bayesian treatment is not tractable. Let us write $\boldsymbol{\Theta}$ the set of variables which will be mode-estimated. Note that they do not need to be exactly parameters in the sense it is often understood in EM, *i.e.* they can be given a prior probability. We are still writing $\mathbf{Z}$ for the other latent variables and parameters which are treated with a variational approach. The full model likelihood is

$$\begin{align*}
    p(\mathbf{X}, \mathbf{Z}, \boldsymbol\Theta)
    & = p(\mathbf{Z}, \boldsymbol\Theta \mid \mathbf{X})~p(\mathbf{X}) \\
    & = p(\mathbf{Z} \mid \boldsymbol\Theta, \mathbf{X})~p(\boldsymbol\Theta \mid \mathbf{X}) p(\mathbf{X}) \\
    & = p(\mathbf{Z} \mid \boldsymbol\Theta, \mathbf{X})~p(\boldsymbol\Theta, \mathbf{X}) \\
\end{align*}$$

We can write as before

$$\begin{align*}
    \ln p(\boldsymbol\Theta, \mathbf{X})
    & = \ln p(\boldsymbol\Theta, \mathbf{X}) \int q(\mathbf{Z})\mathrm{d}\mathbf{Z} \\
    & = \mathcal{L}(q) + \mathrm{KL}\left( q \middle\| p \right)
\end{align*}$$

We are thus looking for $q(\mathbf{Z})$ that approximates $p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol\Theta)$. If $\boldsymbol\Theta$ is fixed, this is similar to the previous variational framework, where we would be looking for $q_{\boldsymbol\Theta}(\mathbf{Z})$ that approximates $p_{\boldsymbol\Theta}(\mathbf{Z} \mid \mathbf{X})$. Additionally, for any current posterior $q_{\boldsymbol\Theta}$, we can update the mode-parameter $\boldsymbol\Theta$ by choosing the value that maximises the lower bound $\mathcal{L}\_\boldsymbol{\Theta}(q_\boldsymbol{\Theta})$.

***

*Created by Yaël Balbastre on 6 April 2018. Last edited on 6 April 2018.*

***
