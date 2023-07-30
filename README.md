# OneHotGraph

Implementation of a graph neural network where each node has a unique one hot encoding as addition to the usual vector of each node. This should help overcome the limitations of WL Isomorphism test. This was a project of mine during a research oriented internship at the institute of machine leanring of the Johannes Kepler University in Linz.

This repository includes a utils folder that includes most of the code that was written. Moreover, there are 5 jupyter notebooks which were used by me to experiment, however those are not cleaned for other users but may still contain some valueables.

Mainly the novelty of this work lies in the Attention One Hot Graph (AOHG) and the Isomorphism One Hot Graph (IOHG) that both have some unique properties. Moreover the mini-batching techinque that I used is slightly more memory consuming but faster as the one used per default with pytorch geoemtric. For a more detailed description see the corrsponding report OneHotGraph_Lab_Report.pdf

If you use anything from this repository than please cite

```bibtex
@misc{krauck_onehotgraph_2023,
    author = {Krauck, Alexander},
    title = {One Hot Graph},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/alexanderkrauck/OneHotGraph}},
}
```
