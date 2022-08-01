---
title: 'Pokeminos: a Python package for creation of 3D volumes containing polyominoes'
tags:
  - Python
  - computer vision
  - polyominos
  - data simulation
authors:
  - name: Anna A. Kotanska
    orcid: 0000-0001-6377-5477
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Mark Basham with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Rosalind Franklin Institute, UK
   index: 1
 - name: Department of Chemistry, Oxford Univeristy, UK
   index: 2
date: 13 August 2017
# bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Pokemoinoes is a toy model for creation of 3D Volumes containing geometrical 
objects (polyominoes) for applications in computer vision. 

# Statement of need

Statement of need will live here. 
I am now testing creation of a PDF based on this paper.md file. 

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure](figures/isomap3d.png){ width=20% }
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:

![Caption for example figure.]
(figures/isomap3d.png){ width=20% }

# Acknowledgements

We acknowledge contributions from...

# References
