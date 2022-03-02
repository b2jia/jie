Jie
---

``Jie`` is a spatial genome aligner. This package parses true 
chromatin imaging signal from noise by aligning signals to a 
reference DNA polymer model.

The codename is a tribute to the Chinese homophones:

- ``结 (jié)`` : a knot, a nod to the mysterious and often entangled structures of DNA
- ``解 (jiĕ)`` : to solve, to untie, our bid to uncover these structures amid noise and uncertainty
- ``姐 (jiĕ)`` : sister, our ability to resolve tightly paired replicated chromatids

Installation
------------
Step 1 - Clone this repository::

    git clone https://github.com/b2jia/jie.git
    cd jie
Step 2 - Create a new conda environment and install dependencies::

    conda create --name jie -f environment.yml
    conda activate jie
    
Step 3 - Install ``jie``::

    pip install -e .
    
To test, run::

    python -W ignore test/test_jie.py

Usage
-----
``jie`` is an exposition of chromatin tracing using polymer physics. The main function of this package is to 
illustrate the utility and power of spatial genome alignment.

``jie`` is NOT an all-purpose spatial genome aligner. Chromatin imaging is a nascent field and data collection is still being standardized. This aligner may not be compatible with different imaging protocols and data formats, among other variables.

We provide a vignette under ``jie/jupyter/``, with emphasis on ``inspectability``. This walks through the intuition of our spatial genome alignment and polymer fiber karyotyping routines::

    00-spatial-genome-alignment-walk-thru.ipynb

We also provide a series of Jupyter notebooks (``jie/jupyter/``), with emphasis on ``reproducibility``. This reproduces figures from our accompanying manuscript::

    01-seqFISH-plus-mouse-ESC-spatial-genome-alignment.ipynb
    02-seqFISH-plus-mouse-ESC-polymer-fiber-karyotyping.ipynb
    03-seqFISH-plus-mouse-brain-spatial-genome-alignment.ipynb
    04-seqFISH-plus-mouse-brain-polymer-fiber-karyotyping.ipynb
    05-bench-mark-spatial-genome-agignment-against-chromatin-tracing-algorithm.ipynb  

A command-line tool forthcoming. 

Motivation
----------

Multiplexed DNA-FISH is a powerful imaging technology that enables us to
peer directly at the spatial location of genes inside the nucleus. Each gene appears as tiny dot under imaging. 

Pivotally, figuring out which dots are physically linked would trace out the structure of chromosomes. 
Unfortunately, imaging is noisy, and single-cell biology is extremely variable. 
The two confound each other, making chromatin tracing prohibitively difficult!

For instance, in a diploid cell line with two copies of a gene we expect to see two spots. 
But what happens when we see:

- ``Extra signals``: 
    - Is it ``noise``?
        -  ``Off-target labeling``: The FISH probes might inadvertently label an off-target gene
    - Or is it ``biological variation``?
        - ``Aneuploidy``: A cell (ie. cancerous cell) may have more than one copy of a gene
        - ``Cell cycle``: When a cell gets ready to divide, it duplicates its genes    
- ``Missing signals``: 
    - Is it ``noise``?
        -  ``Poor probe labeling``: The FISH probes never labeled the intended target gene
    - Or is it ``biological variation``?
        - ``Copy Number Variation``: A cell may have a gene deletion

If true signal and noise are indistinguishable, how do we know we are selecting 
true signals during chromatin tracing? It is not obvious which spots should be connected
as part of a chromatin fiber. This dilemma was first aptly characterized by Ross et al.
(https://journals.aps.org/pre/abstract/10.1103/PhysRevE.86.011918), which is nothing
short of prescient...!

``jie`` is, conceptually, a spatial genome aligner that disambiguates spot 
selection by checking each imaged signal against a reference polymer physics 
model of chromatin. It relies on the key insight that the ``spatial separation``
between two genes should be ``congruent`` with its ``genomic separation``.

It makes no assumptions about the expected copy number of a gene, and when 
it traces chromatin it does so instead by evaluating the ``physical likelihood`` 
of the chromatin fiber. In doing so, we can uncover copy number variations and 
even sister chromatids from multiplexed DNA-FISH imaging data.

Contact
-------

:Author: Bojing (Blair) Jia
:Email: b2jia at eng dot ucsd dot edu
:Position: MD-PhD Student, Ren Lab

For other work related to single-cell biology, 3D genome, and chromatin imaging, please visit Prof. Bing Ren's website: http://renlab.sdsc.edu/

For a spatial genome aligner that leverages the forwards-backwards algorithm to estimate the most likely state at each 
locus, please visit: https://github.com/heltilda/align3d