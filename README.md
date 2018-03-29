# Notes on Bayesian models for Computational Anatomy

This website is aimed at keeping trace of notes on different modelling approaches that are used in unified models of segmentation and registration. I've tried to break down the model in small pieces in order to be able to focus on the aim and practical implementation of each variable (it also leads to less cluttered notations). Obviously, when everyting is put together in the same model, there are some additional issues that arise ; in particular some variables that are assumed observed in the small pieces become latent variable, with a posterior distribution and thus uncertainty that must be taken into account.

## Implementation

I am using a GitHub page and its implicit [Jekyll](https://jekyllrb.com) machinerie to convert markdown files to webpages. The layout is a slightly modified version of [Lanyon](https://github.com/poole/lanyon). I've also hadded [Mathjax](https://www.mathjax.org) support in order to render equations.

## How to modify pages?

Just use GitHub's editor to modify the markdown pages. Because of Mathjax, a few points must be kept in mind to avoid strange behaviours:
- Underscores (`_`) are used both in Tex formulas (to indicate underscripts: `x_i`) and in Markdown (to indicate emphasis: `_important_`). It is thus advised to rather use asteriks (`*`) rather than underscores to indicate emphasis in Markdown (`*important*`).
- Use double dollars (`$$x = 0$$`) around displayed (*i.e.*, centered) equations and single dollars (`$x = 0$`) around inine equations.
- As a result, avoid using dollars in your text! If it is necessary, you should escape them (`\$`).
- Avoid using a veritcal bar (`|`) in math mode. This is quite annoying as it is needed for KL divergences and (some) conditional probabilities. I haven't found a solution yet.

## How to add pages

For now, the organisation is quite messy. Just add a markdown file at the root, with the following front matter:
```
---
layout:  page
title:   My title
mathjax: true
---
```
Is mathjax is not needed in the file, use `mathjax: false`. In this case, no need to escape dollars. Note that such a page wil find its way in the sidebar panel. If you do not want the page to appear in the sidebar, use `layout: default`.

## License

&copy; 2018 Yaël Balbastre

This content has an educational purpose (in practice, my own education is the purpose :p). It is licensed under Creative Commons BY-NC. As a result, you may use, modify and redistribute it for non-commercial purposes, as long as I am acknowledged.

[TL;DR: CC BY-NC](https://tldrlegal.com/license/creative-commons-attribution-noncommercial-4.0-international-(cc-by-nc-4.0))
