---
layout:  page
title:   Hierarchical Gaussian mixture
mathjax: true
---

Context
-------

Gaussian mixtures are classically used to model multi-modal data. In most state-of-the-art image analysis methods, they are the cornerstone of the segmentation component. Typically, in medical images, voxels are assumed to belong to a finite number of classes, which all have a particular intensity distribution modeled by a Gaussian. Since both class labelling and Gaussian parameters (mean and covariance) are unknown, methods usually rely on an Expectation-Maximisation scheme to iterate between computing the expected labels and computing the Gaussian parameters that maximise the model likelihood. The problem can be tackled in a more Bayesian way by placing a Normal-Wishart prior over the prameters of each Gaussian, in which case [variational inference]({{site.baseurl}}/variational-inference) is used. This allows to regularise the parameter estimation by penalising unlikely values. Additionnaly, if a set of several images is segmented at once, it may be of interest to also optimise the Normal-Wishart prior parameters by, *e.g.*, maximising the model evidence.

In two cases, it might be interesting to place an Inverse-Wishart prior on the Wishart prior parameters:

1. If all classes are expected to possess roughly similar variances, the Inverse-Wishart prior could be shared by all (prior) covariance matrices. This would penalise covariance matrices that deviate from the mean.

   > [Constrained Gaussian mixture]({{site.baseurl}}/constrained-gaussian-mixture)

2. If images were acquired in different centers, or with different sequences, it makes sense to model this by associating a Normal-Wishart prior with each center/sequence and placing a shared Normal-Inverse-Wishart prior over all their parameters.

   > [~~Multicentre Gaussian mixture~~]({{site.baseurl}}/multicentre-gaussian-mixture)

***

*Created by Yaël Balbastre on 11 April 2018. Last edited on 13 April 2018.*

***
