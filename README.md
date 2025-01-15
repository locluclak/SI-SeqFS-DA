# Statistical Inference for Sequential Feature Selection after Domain Adaptation

This package has the following requirements:

    numpy
    mpmath
    matplotlib
    scipy
    sklearn
    statsmodels

We recommend to install or update anaconda to the latest version and use Python 3 (We used Python 3.11.5).

**Note**: This package utilizes the scipy package to solve linear programming problems using the simplex method. However, the default scipy package does not provide the set of basic variables in its output. To address this limitation, we include a custom-modified version, scipy_, which extends the functionality to return the set of basic variables.