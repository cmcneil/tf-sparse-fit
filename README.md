# Sparse Fitting in Python with Tensorflow


## Intro
When fitting very high-dimensional linear models without much data, it is often necessary to impose a sparse prior on the weights of the model to avoid overfitting. Sparse fits can also be desireable in some situations to increase interpretability of the model, or computational efficiency.

Unlike [ordinary least-squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)(OLS) regression, or even OLS with [Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization), there is not a simple, closed form solution to most sparse priors. Therfore, we're stuck with iterative solutions. For very large datasets (in either number of points or number of dimensions), this becomes slow *very* quickly. To address this need, I wrote a couple of simple fitting functions implemented in [Tensorflow](https://www.tensorflow.org/). As Tensorflow graphs can run on GPU, this was an easy way to get a GPU implementation. Another advantage is that the script will generate a `log` directory. Point Tensorboard at this log directory to see very detailed progress for slow fits.

This package is really trying to provide ease of use with a GPU implementation. If you aren't into Python, or if you aren't going to use the GPU version of Tensorflow, I don't know if the overhead is worth it versus other CPU-based sparse fits, for example using Coordinate Descent with the [glmnet](https://github.com/cran/glmnet) package.

## Dependencies
This package depends on Tensorflow. See the Tensorflow [Installation guide](https://www.tensorflow.org/get_started/os_setup) on how to install for your system. This package does need CUDA, but not CUDNN. Make sure to get the GPU version of Tensorflow.

It also depends on `scipy` and `numpy`:

    sudo pip install scipy numpy
    
## Use
This library provides two different fitting algorithms (personally, I'm a fan of Threshold Gradient Descent):


### Group LASSO
LASSO (least absolute shrinkage and selection operator) is simply optimizing an objective with both least squares and an L1 penalty on the weights. This is equivalent to imposing a Laplace prior distribution on your weights. Group LASSO is a modification permitting the choice of a different regularization parameter for each dimension. ([Yuan et al, 2006](http://pages.stat.wisc.edu/~myuan/papers/glasso.final.pdf))

From the `group_lasso_fit` docstring:
```python
    """Run group a LASSO fit using stochastic gradient descent on GPU.

        Parameters
        ----------
        X_tr:
            The training set. A numpy array of shape (time_points, #features).
        Y_tr:
            The training labels. A numpy array of shape (time_points, #labels)
            (The labels might be BOLD signal across different voxels,
            for example.)
        X_test:
            The test set. A numpy array of shape (time_points, #features). The
            number of time points may be different between the training and
            test sets, but the number of features may not.
        Y_test:
            The test labels. A numpy array of shape (time_points, #labels)
        batch_size: int, 100
            The minibatch size for the stochastic gradient descent.
        train_dir: str, '.'
            The directory that model and log directories will be saved in.
            This directory is important if you want to view the fit in
            Tensorboard.
        max_iterations: int, 350
            Early stopping is not yet implemented. Right now, you simply set
            a number of iterations.
        learning_rate: float, 0.0001
            The SGD learning rate.
        l1_params:
            A numpy array with length equal to the number of features, that
            specifies L1 regularization constants for every feature.

        Returns
        -------
        A dictionary containing both 'weights' and 'predictions', which are
        numpy arrays containing the learned weights, and the pearson
        correllation for each label, respectively.
        """
```
To clarify the `l1_params` argument, if you want to do normal LASSO (instead of Group LASSO), just pass a list (of length equal to the number of variables/features/coefficients) all with the same float.

### Threshold Gradient Descent
Threshold gradient descent ([Friedman and Popescu, 2004](http://www.stat.washington.edu/courses/stat527/s13/readings/FriedmanPopescu2004.pdf)), is a fitting technique that regularizes by thresholding the gradients during a gradient descent, instead of modifying the loss function, which remains as the least-squares error. For each gradient step, TGD finds the maximum gradient, and then only allows gradients of a certain size relative to the maximum to pass. I recommend using it only on Z-scored data for stability of the gradient process.

Threshold gradient descent has similar training trajectories to LASSO, and produces Laplace-like weight distribution, but is less well characterized than LASSO. It seems to perform well in practice, and the hyperparameter space (the threshold) is more regular.

From the `threshold_gradient_descent_fit` docstring:

```python
    """Run Threshold gradient descent on GPU.

        Parameters
        ----------
        X_tr:
            The training set. A numpy array of shape (time_points, #features).
        Y_tr:
            The training labels. A numpy array of shape (time_points, #labels)
            (The labels might be BOLD signal across different voxels,
            for example.)
        X_test:
            The test set. A numpy array of shape (time_points, #features). The
            number of time points may be different between the training and test
            sets, but the number of features may not.
        Y_test:
            The test labels. A numpy array of shape (time_points, #labels)
        batch_size: int, 100
            The minibatch size for the stochastic gradient descent.
        train_dir: str, '.'
            The directory that model and log directories will be saved in.
            This directory is important if you want to view the fit in
            Tensorboard.
        max_iterations: int, 350
            Early stopping is not yet implemented. Right now, you simply set
            a number of iterations.
        learning_rate: float, 0.0001
            The SGD learning rate.
        threshold_grad_desc: float, 0.5
            A float between 0 and 1 specifying how to threshold the gradients.
            A value of 0.dd will only allow gradients of magnitude that are
            dd% of the maximum gradient's absolute value.
        use_active_set: bool, True
            Specifies whether or not to use an active set while fitting.
        verbose: bool, True
            Specifies whether or not to use

        Returns
        -------
        A dictionary containing both 'weights' and 'predictions', which are
        numpy arrays containing the learned weights, and the pearson
        correllation for each label, respectively.
        """
  ```
  
The `use_active_set` parameter controls whether or not to maintain an active set of gradients that are always allowed to move. Using an active set means that the gradient for a particular dimension only needs to pass threshold once. This makes sense if you think about TGD as a feature selection process. Once we've already decided a weight will be non-zero, we might as well allow it to tune itself more finely.
  
## TODO
* Make larger-than-memory code available for large datasets.
* Add early stopping, with the option of a separate early-stopping set.
* Add variable learning rate, dependent on stopping set error trajectory.

## Contribute
Feel free to file issues and make pull requests. Hope it's useful!
