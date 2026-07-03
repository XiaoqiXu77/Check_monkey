# Check_monkey

Code accompanying the paper:

**Prefrontal mechanisms of goal progress inference and monitoring in macaque monkeys**

Xiaoqi Xu, Frederic M Stoll, Matteo di Volo, Charles R. E. Wilson, Emmanuel Procyk, Nils Kolling

bioRxiv 2026.05.19.726190; doi: https://doi.org/10.64898/2026.05.19.726190


## Overview

This repository contains the code used for the the behavioural and neural analyses in **Prefrontal mechanisms of goal progress inference and monitoring in macaque monkeys**.

## Repository structure

```text
.
├── utils/              # Helper functions
├── bayesian_model.m    # Bayesian inference of progress rate
├── behaviour.m         # Behavioural analyses
├── decoding.m          # Decoding various task variables
├── fb_geometry.m       # Cross-condition decoding and cosine similarity of feedback valence axes across rates
├── pca_analysis/       # PCA plot of feedback responses under different rates
├── preprocessing.m     # Data preprocessing
├── rnn_model.py        # Recurrent neural network model
├── time_scales_corr.m  # Correlation between each neuron’s timescale and the magnitude of its progress-rate encoding
└── README.md
```

## Requirements

* Python
* PyTorch
* Matlab


## Data

The dataset used in this work is available upon request.


## Citation

If you use this code, please cite:

```bibtex
@article {Xu2026.05.19.726190,
	author = {Xu, Xiaoqi and Stoll, Frederic M and di Volo, Matteo and Wilson, Charles R. E. and Procyk, Emmanuel and Kolling, Nils},
	title = {Prefrontal mechanisms of goal progress inference and monitoring in macaque monkeys},
	elocation-id = {2026.05.19.726190},
	year = {2026},
	doi = {10.64898/2026.05.19.726190},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}

```


## Contact

For questions about the code or paper, contact Xiaoqi Xu at xiaoqi.xu at inserm.fr.
