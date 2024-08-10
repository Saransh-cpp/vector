---
title: "Vector: arrays of 2D, 3D, and Lorentz vectors"
tags:
  - Python
  - vector algebra
  - high energy physics
authors:
  - name: Henry Schreiner
    orcid: 0000-0002-7833-783X
    equal-contrib: true
    affiliation: 1
  - name: Jim Pivarski
    orcid: 0000-0002-6649-343X
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Saransh Chopra
    orcid: 0000-0003-3046-7675
    equal-contrib: true
    affiliation: 1

affiliations:
  - name: Princeton University
    index: 1
date: 10 August 2024
bibliography: paper.bib
---

# Summary

Vector algebra is a crucial component of data analysis pipelines in high energy
physics, enabling physicists to transform raw data into meaningful results that
can be visualized. Given that high energy physics data is not uniform, the
vector algebra frameworks or libraries are expected to work readily on
non-uniform or jagged data, allowing users to perform operations on an entire
jagged array in minimum passes. Furthermore, optimizing memory usage and
processing time has become essential with the increasing computational demands
at the LHC. Vector is a Python library for creating and manipulating 2D, 3D,
and Lorentz vectors, especially arrays of vectors, to solve common physics
problems in a NumPy-like way. The library enables physicists to operate on high
energy physics data in a high level language without compromising speed.
The library is already in use at LHC and is a part of frameworks, like Coffea,
employed by physicists across multiple high energy physics experiments.

# Statement of need

Vcetor is currently the only Lorentz vector library providing a Pythonic
interface but a C++ (through Awkward) computational backend. Vector integrates
seamlessly with the existing high energy physics ecosystem and the broader
scientific Python ecosystem, including libraries like Dask and Numba. The
library implements a variety of backends for several purposes. Although
vector was written with high energy physics in mind, it is a general-purpose
library that can be used for any scientific or engineering application. The
library houses 3+2 numerical backends for experimental physicists and 1 symbolic
backend for theoretical physicists. These backends include a pure Python object
backend for simple computations, a SymPy backend for symbolic computations, a
NumPy backend for computations on regular data, an Awkward backend for
computations on jagged data, and implementations of the Object and the Awkward
backend in Numba for just-in-time compilable operations. Support for JAX and
Dask is also provided through the Awkward backend, which enable vector
functionalities to support automatic differentiation and parallel computing.

## Impact

Vector has become the de facto library for vector algebra in Python based high
energy physics data analysis pipelines. The library has been installed over
2 million times and 314 GitHub repositories use it as a dependency at the time
of writing this paper. Along with being utilized directly in analysis pipelines
at LHC [@Kling_2023:2023; @Held:2024], the library is also used by other user-facing libraries, such as,
Coffea, MadMiner [@Brehmer:2020], FastJet, Spyral, Weaver, and pylhe. The
library is also used
in multiple teaching materials for graduate courses and workshops. Finally,
given the generic nature of the library, it is also often used in non high
energy physics use cases.

# Acknowledgements

The work on vector was supported by NSF cooperative agreements OAC-1836650 (IRIS-HEP) and PHY-2323298 (IRIS-HEP). We would also like to thank the
contributors of vector and the Scikit-HEP community for their support.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:

- `@author:2001` -> "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Reference
