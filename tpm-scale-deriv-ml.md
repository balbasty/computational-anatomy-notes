---
layout:  default
title:   Scaling tissue probability maps - Derivations
mathjax: true
---

Scaling tissue probability maps - Derivations
=============================================

This section refers to the main article [Scaling tissue probability maps](/tpm-scale).
Let's use Matlab's symblic toolbox to differentiates our objective functions.

Maximum Likelihood
------------------

Remember that, keeping only terms that depend on the scaling weights, the log-likelihood is:

$$\mathcal{L}
= \sum_{k=1}^K \left[ \left(\sum_{i=1}^I z_i^{(k)}\right)\ln w_k \right] - \sum_{i=1}^I \left[ \left(\sum_{k=1}^K z_i^{(k)}\right) \ln\left(\sum_{l=1}^Kw_l\mu_i^{(l)}\right) \right]$$

If we write the sum of labels over all voxels as $\overline{\mathbf{z}} = \sum_{i=1}^I z_i^{(k)} \in \mathbb{R}^K$ and the sum of labels over classes as $\mathbf{s} = \sum_{k=1}^K z_i^{(k)} \in \mathbb{R}^I$, this objective function can be written as:

$$\mathcal{L}
= \overline{\mathbf{z}}^{\mathrm{T}} \ln \mathbf{w} - \sum_{i=1}^I s_i \ln\left(\mathbf{w}^{\mathrm{T}}\boldsymbol{\mu}_i\right)$$

Here's a code snippet allowing to "guess" the gradient and Hessian of the objective function with respect to the scaling weights. We write a very basic exemple with 3 classes and 2 voxels.


First, declare the symbolic variables. Note that we take into account $\sum_{k=1}^K z_i^{(k)}$, even though it should usually sum to one.
```=
w  = sym('w',  [3,1]);     % weights
z  = sym('z',  [3,1]);     % sum of binary labels over voxels
m1 = sym('m1', [3,1]);     % TPM at the first voxel
m2 = sym('m2', [3,1]);     % TPM at the second voxel
syms s1                    % sum of all labels at the first voxel
syms s2                    % sum of all labels at the second voxel
assume(w, 'positive')
```

Now, let's write the **negative**[^negativell] log-likelihood and differentiate:

[^negativell]: We use the negative log-likelihood so that the optimisation problem is a **minimisation** problem. Both are equivalent: we place ourselves in a minimisation framework by convention.

```=+
L = - z.' * log(w) + s1*log(w.'*m1) + s2*log(w.'*m2);

pretty(simplify(diff(L, sym('w1'))))               % gradient
```
```
  z1            m11 s1                     m21 s2
- -- + ------------------------ + ------------------------
  w1   m11 w1 + m12 w2 + m13 w3   m21 w1 + m22 w2 + m23 w3
```
```=+
pretty(simplify(diff(L, sym('w1'), sym('w1'))))    % Hessian (diagonal)
```
```
                   2                             2
 z1             m11  s1                       m21  s2
--- - --------------------------- - ---------------------------
  2                             2                             2
w1    (m11 w1 + m12 w2 + m13 w3)    (m21 w1 + m22 w2 + m23 w3)
```
```=+
pretty(simplify(diff(L, sym('w1'), sym('w2'))))    % Hessian (off-diagonal)
```
```
           m11 m12 s1                    m21 m22 s2
- --------------------------- - ---------------------------
                            2                             2
  (m11 w1 + m12 w2 + m13 w3)    (m21 w1 + m22 w2 + m23 w3)
```

We guess that the gradient can be written as:
$$\mathbf{g} = \sum_{i=1}^I \left(\sum_{k=1}^K z_i^{(k)}\right) \frac{\boldsymbol{\mu}_i}{\boldsymbol{\mu}_i^{\mathrm{T}} \mathbf{w}} - \mathrm{diag}(\mathbf{w})^{-1}\left(\sum_{i=1}^I\mathbf{z}_i\right)$$

and the Hessian as:
$$\mathbf{H} = \mathrm{diag}\left(\mathrm{diag}(\mathbf{w})^{-2}\left(\sum_{i=1}^I\mathbf{z}_i\right)\right) - \sum_{i=1}^I \left(\sum_{k=1}^K z_i^{(k)}\right) \frac{\boldsymbol{\mu}_i \boldsymbol{\mu}_i^{\mathrm{T}}}{\left(\boldsymbol{\mu}_i^{\mathrm{T}} \mathbf{w}\right)^2}$$

Let us check that with the symbolic toolbox. The equivalent Matlab code is:
```=
g = -z./w + s1*m1/(m1.'*w) + s2*m2/(m2.'*w);
H = diag(z./(w.^2)) - s1*(m1*m1.')/(m1.'*w)^2 - s2*(m2*m2.')/(m2.'*w)^2;

```
And if we compare with the automatic differentiation:
```=+
check_g1  = simplify(g(1) - diff(L, sym('w1')), 1000)
```
```
check_g1 =

0
```
```=+
check_h11 = simplify(H(1,1) - diff(L, sym('w1'), sym('w1')), 1000)
```
```
check_h11 =

0
```
```=+
check_h12 = simplify(H(1,2) - diff(L, sym('w1'), sym('w2')), 1000)
```
```
check_h12 =

0
```

You could ask if a closed form solution to $\mathbf{g} = \mathbf{0}$ exists, which would make the Hessian unneeded. Let us try to find a solution with the symbolic toolbox:
```=+
solve(g, w)
```
```
ans =

  struct with fields:

    w1: [0×1 sym]
    w2: [0×1 sym]
    w3: [0×1 sym]
```
Even in this very simple case, we cannot find a solution in closed form. As explained before, two solutions exist, either by keeping the denominator fixed, or by using a numerical optimisation procedure such as Gauss-Newton.

---
Created by Yaël Balbastre on 27 March 2018.
Last edited on 28 March 2018.
