# Exploring Dimensionality Reduction of SDSS Spectral Abundances

![2-dimensional representations](https://github.com/user-attachments/assets/75950eef-773b-425b-b0f1-f9cf00084f9e)

## Abstract
High-resolution stellar spectra offer valuable insights into atmospheric parameters and chemical compositions. However, their inherent complexity and high-dimensionality present challenges in fully utilizing the information they contain. In this study, we utilize data from the Apache Point Observa- tory Galactic Evolution Experiment (APOGEE) within the Sloan Digital Sky Survey IV (SDSS-IV) to explore latent representations of chemical abundances by applying five dimensionality reduction techniques: PCA, t-SNE, UMAP, Autoencoder, and VAE. Through this exploration, we evaluate the preservation of information and compare reconstructed outputs with the original 19 chemical abun- dance data. Our findings reveal a performance ranking of PCA < UMAP < t-SNE < VAE < Au- toencoder, through comparing their explained variance under optimized MSE. The performance of non-linear (Autoencoder and VAE) algorithms has approximately 10% improvement compared to lin- ear (PCA) algotirhm. This difference can be referred to as the ”non-linearity gap.” Future work should focus on incorporating measurement errors into extension VAEs, thereby enhancing the reliability and interpretability of chemical abundance exploration in astronomical spectra.

## Key Words
Dimensionality Reduction; Chemical Abundances, APOGEE, Neural Network

## Dataset
https://www.sdss4.org/dr17/data_access/
