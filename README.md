# MU-NeRF: Multi-Scale Uncertainty-Aware Neural Randiance Fields

**🚧 Work in Progress**

This repository is currently under active development. The core codebase is functional, but some components remain unpolished or redundant. A cleaned and fully documented release will follow.
Please reach out if anything is unclear: bas.verkennis.code@gmail.com

# Overview
This repository contains the implementation of MU-NeRF, an extension of ActiveNeRF that introduces multi-scale uncertainty-aware supervision for improved neural radiance field reconstruction under sparse and uncertain data conditions.

The code also supports the original ActiveNeRF (Pan et al., 2022), enabling comparative experiments. The repository also features evaluation scripts and an intuitive GUI for qualitative assessment.

## Installation

```
git clone https://github.com/basverkennis/MU-NeRF
cd MU-NeRF
pip install -r requirements.txt
```

# Getting Started
To train the MU-NeRF and ActiveNeRF models, run the provided training script:

bash training.sh

# Citation & Acknowledgement
This project builds upon the implementation of ActiveNeRF:

Pan, X., Lai, Z., Song, S., & Huang, G. (2022).
ActiveNeRF: Learning Where to See with Uncertainty Estimation.
In Avidan, S., Brostow, G., Cissé, M., Farinella, G.M., & Hassner, T. (Eds.), Computer Vision – ECCV 2022, Lecture Notes in Computer Science, vol 13693. Springer, Cham.
https://doi.org/10.1007/978-3-031-19827-4_14
