# OneHotGraph

Research internship artifact exploring one-hot node identity features in graph neural networks.

Author: Alexander Krauck  
Context: Institute of Machine Learning, Johannes Kepler University Linz  
Focus: graph neural networks, graph isomorphism, graph representation learning

## What This Is

This repository contains experimental code from a research-oriented internship at the Institute of Machine Learning at Johannes Kepler University Linz.

The project explored whether adding node-specific one-hot identity features can change the behavior of graph neural networks on graph-structure tasks. The work was exploratory and should be read as a research artifact, not as a maintained package.

## Research Question

Standard message-passing GNNs have known limitations related to the Weisfeiler-Lehman graph isomorphism test. This project explored whether adding unique node identity features could help distinguish certain graph structures and improve expressiveness in selected settings.

The experiments focused on:

- one-hot node identity features
- graph isomorphism-related behavior
- attention-based graph models
- batching and memory tradeoffs
- exploratory comparison against standard graph neural network setups

## Repository Structure

```text
OneHotGraph/
├── utils/                    # core experiment code
├── grids/                    # experiment/grid configurations
├── main.py                   # experiment entry point
├── environment.yml           # conda environment
├── OneHotGraph_Lab_Report.pdf
├── citation.bib
└── *.ipynb                   # exploratory notebooks
```

## Notebooks

The notebooks are preserved as part of the research process. They are not cleaned as a tutorial or library API.

```text
00_TryStuff.ipynb
01_Data_Exploration.ipynb
02_Pytorch_Geometric_Tryout.ipynb
03_SparseTrying.ipynb
04_OGH_Experiments.ipynb
```

## Setup

```bash
conda env create -f environment.yml
conda activate onehotgraph
```

Run experiments:

```bash
python main.py
```

## Report

For more context, see:

```text
OneHotGraph_Lab_Report.pdf
```

## Citation

```bibtex
@misc{krauck2023onehotgraph,
  author = {Krauck, Alexander},
  title = {OneHotGraph},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/alexanderkrauck/OneHotGraph}}
}
```

## Status

Exploratory research artifact. Kept public for transparency, but not maintained as a reusable framework.
