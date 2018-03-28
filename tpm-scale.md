---
layout: default
title:  Scaling tissue probability maps
---
{% include lib/mathjax.html %}

Scaling tissue probability maps
===============================

Context
-------

Tissue probability maps (TPMs), or _templates_, encode the prior probability of each voxel in an image to belong to one of \\(K\\) classes. Typically, they reflect the proportion of subjects, in the general population, that show a given class in a given location. These TPMs play a crucial role in segmentation models as they provide *local* prior information about class probability, thus helping regularise the segmentation in an anatomical way. TPMs are aligned to each observed brain and deformed to match as well as possible their anatomy[^registration]. The following figure shows a brain image along with the true labels and the deformed template (each color represents one class, mixed colors correspond to class probabilities):

[^registration]: We do not tackle the issue of deforming TPMs in this article. It may be done beforehand or, more consistantly, at the same time as the segmentation in an iterative fashion.

In the best case scenario, the deformed TPMs would match almost exactly the true labels. However, because deformations are not entirely _free_ (_i.e._, they must be smooth), there is often some kind of leeway, especially if the observed image possess an unusual anatomy (as might be the case if it is pathological). In this case, an issue is that the global expected class proportions in the template might be different from the actual class proportion, and thus bias the segmentation. For exemle, in the above exemple, the expected tissue proportions, are [x y z] whereas the true proportion is [x' y' z']. Consequently, it is necessary to modulate the TPMs to change the global expected class proportion.

Formal writing
--------------

Let us write the true labels as a series of binary images, each acting as a mask of voxels belonging to one class. In vector form, this yield: \\(\mathbf{z} = \left[z_i^{(k)} \in \left\\{0,1\right\\} \mid  i \in 1\dots I ~;~ k \in 1\dots K \right]\\), where \\(i\\) indexes voxels and \\(k\\) indexes classes. In a similar fashion, we write the template as a series of probability images: \\(\boldsymbol{\mu} = \left[\mu_i^{(k)} \in \left[0,1\right] \mid  i \in 1\dots I ~;~ k \in 1\dots K \right]\\). The idea is to find a series of factors, \\(w_k\\), so that prior probabilities become:

$$p_i^{(k)} = \frac{w_k\mu_i^{(k)}}{\sum_{l=1}^K w_l\mu_i^{(l)}}$$

Formally, saying that these are probabilities of belonging to each class is equivalent to saying that they are parameters of a _Categorical_ distribution, from which stem the actual labels. We can thus write the conditonal probability:

$$\forall i \in 1\dots I,~ p(\mathbf{z}_i \mid \boldsymbol{\mu}_i, \boldsymbol{w}) = \mathcal{C}\left(\mathbf{z}_i ~\middle|~ \mathbf{p}_i \right) = \prod_{k=1}^K \left(p_i^{(k)}\right)^{z_i^{(k)}}$$

The conditional log-likelihood of the complete image can be written as:

$$\begin{split}\mathcal{L} 
& = \sum_{i=1}^I \sum_{k=1}^K \left[ z_i^{(k)}\left(\ln w_k + \ln \mu_i^{(k)} - \ln\left(\sum_{l=1}^Kw_l\mu_i^{(l)}\right)\right) \right] \\
& = \sum_{k=1}^K \left[ \left(\sum_{i=1}^I z_i^{(k)}\right)\ln w_k  + \sum_{i=1}^I z_i^{(k)} \ln\mu_i^{(k)} \right] - \sum_{i=1}^I \left[ \left(\sum_{k=1}^K z_i^{(k)}\right) \ln\left(\sum_{l=1}^Kw_l\mu_i^{(l)}\right) \right]
\end{split}$$

Maximum likelhood
-----------------

A first strategy is to find a _maximum likelihood_ (ML) value for the weights, that is, the value that maximises the conditional log-likelihood[^log]:

[^log]: Using the log-likelihood rather than the likelihood has several advantages. First, because the logarithm is strictly monotonic, a maximum of the log-likelihood is a maximum of the likelihood. Second, it transforms products in sums, often easing the differentiation. Third, it is a transform that is used to solve problems with hidden variables, along with the Expectation-Maximisation algorithm, because of its concavity. 

$$\hat{\mathbf{w}} = \arg\!\min_{\mathbf{w}} -\sum_{i=1}^I \ln p(\mathbf{z}_i \mid \boldsymbol{\mu}_i, \boldsymbol{w})$$

Differentiating with respect to the weights yield:

$$\mathbf{g} = \sum_{i=1}^I \left(\sum_{k=1}^K z_i^{(k)}\right) \frac{\boldsymbol{\mu}_i}{\boldsymbol{\mu}_i^{\mathrm{T}} \mathbf{w}} - \mathrm{diag}(\mathbf{w})^{-1}\left(\sum_{i=1}^I\mathbf{z}_i\right)$$

Sadly, this gradient cannot be solved in closed form. A first solution, used in Ashburner's _Unified Segmentation_ ([Ashburner and Friston, 2005](https://doi.org/10.1016/j.neuroimage.2005.02.018 "Persistent link using digital object identifier")), is to assume \\(\boldsymbol{\mu}_i^{\mathrm{T}} \mathbf{w}\\) constant (*i.e.*, it keeps a previous value), yielding:

$$\tilde{\mathbf{w}} = \mathrm{diag}\left(\sum_{i=1}^I\mathbf{z}_i\right)^{-1}\left(\sum_{i=1}^I \left(\sum_{k=1}^K z_i^{(k)}\right) \frac{\boldsymbol{\mu}_i}{\boldsymbol{\mu}_i^{\mathrm{T}} \mathbf{w}}\right)$$

A second solution is to obtain the optimum numerically with a Gauss-Newton optimisation scheme. In this case, we also need the second derivatives of the objective function:

$$\mathbf{H} = \mathrm{diag}\left(\mathrm{diag}(\mathbf{w})^{-2}\left(\sum_{i=1}^I\mathbf{z}_i\right)\right) - \sum_{i=1}^I \left(\sum_{k=1}^K z_i^{(k)}\right) \frac{\boldsymbol{\mu}_i \boldsymbol{\mu}_i^{\mathrm{T}}}{\left(\boldsymbol{\mu}_i^{\mathrm{T}} \mathbf{w}\right)^2}$$

The optimum can then be optained iteratively, according to the update scheme:

$$\mathbf{w}^{(n+1)} = \mathbf{w}^{(n)} - \left[\mathbf{H}^{(n)}\right]^{-1} \mathbf{g}^{(n)}$$

---
Created by Yaël Balbastre on 27 March 2018.
Last edited on 28 March 2018.
