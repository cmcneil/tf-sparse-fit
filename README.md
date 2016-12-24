# Sparse Fitting with Tensorflow

## Intro
When fitting very high-dimensional linear models without much data, it is often necessary to impose a sparse prior on the weights of the model to avoid overfitting. Sparse fits can also be desireable in some situations to increase interpretability of the model, or computational efficiency.

Unlike [ordinary least-squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)(OLS) regression, or even OLS with [Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization), there is not a simple, closed form solution to most sparse priors. Therfore, we're stuck with iterative solutions. For very large datasets (in either number of points or number of dimensions), this becomes slow *very* quickly. To address this need, I wrote a couple of simple fitting functions implemented in [Tensorflow](https://www.tensorflow.org/). As Tensorflow graphs can run on GPU, this was an easy way to get a GPU implementation.

## Dependencies
The primar
