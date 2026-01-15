# Adaptive-CASSI-Reconstruction
This is the code for the paper "Self-supervised adaptive reconstruction with physical and geometric priors for compressive spectral imaging across varying system parameters".
Cite：
@article{DAI2026131091,
title = {Self-supervised adaptive reconstruction with physical and geometric priors for compressive spectral imaging across varying system parameters},
journal = {Expert Systems with Applications},
volume = {308},
pages = {131091},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2026.131091},
url = {https://www.sciencedirect.com/science/article/pii/S0957417426000059},
author = {Haolin Dai and Haoyang Yu and Zhaohui Jiang and Dong Pan and Weihua Gui},
keywords = {Compressive spectral imaging, Spectral image reconstruction, Self-supervised learning, Physics-driven},
abstract = {Reconstruction in snapshot compressive spectral imaging systems is challenging due to variability in optical system parameters such as mask patterns and spectral channels, often limiting the generalization of conventional models. To address this challenge, we propose a self-supervised adaptive reconstruction framework that fundamentally reformulates the reconstruction process. Rather than treating the network as a fixed prior, we view it as a solution representation over a continuous space of system parameters. For each measurement, the network are directly adapted to the specific system parameters, allowing the model to shift from the training domain to the current measurement domain. This instance-specific parameter refinement preserves the expressive capacity learned during supervised pre-training while eliminating the need for full retraining. The adaptation is guided by a physics-based fidelity term consistent with the forward imaging model and regularized by a geometric cycle consistency constraint incorporating system-invariant priors. In addition, a U-shaped architecture with large-kernel convolutional blocks and spatial-spectral attention fusion modules is employed to capture multi-scale spatial and spectral dependencies. Extensive experiments under diverse system parameters demonstrate superior reconstruction quality and adaptability, offering a practical and generalizable solution for high-fidelity hyperspectral reconstruction. (code: https://github.com/dhl98csu/Adaptive-CASSI-Reconstruction)}
}

