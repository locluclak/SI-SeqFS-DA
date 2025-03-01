# Statistical Inference for Sequential Feature Selection after Domain Adaptation

[![PyPI version](https://badge.fury.io/py/si-seqfs-da.svg)](https://badge.fury.io/py/si-seqfs-da)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package provides a statistical inference framework for sequential feature selection (SeqFS) after domain adaptation (DA). It leverages the SI framework and employs a divide-and-conquer strategy to efficiently compute the p-value of selected features. Our method ensures reliable feature selection by controlling the false positive rate (FPR) while simultaneously maximizing the true positive rate (TPR), effectively reducing the false negative rate (FNR).

*For more details, refer to the paper at https://arxiv.org/abs/2501.09933*

This package has the following requirements:

    numpy
    mpmath
    matplotlib
    scipy
    sklearn

We recommend to install or update anaconda to the latest version and use Python 3 (We used Python 3.11.5).

**Note**: This package utilizes the scipy package to solve linear programming problems using the simplex method. However, the default scipy package does not provide the set of basic variables in its output. To address this limitation, we slightly modified the package so that it can return the set of basic variables by replacing the two files '_linprog.py' and '_linprog_simplex.py' in scipy.optimize module with our modified files in the folder 'files_to_replace'.


### How to Automatically Replace the Files  
If Using Anaconda
- First, activate the target environment:  
```sh
conda activate your-env-name
```
- Then, run the following command to replace the necessary files:

``` sh
python replace_scipy_linprog.py --env anaconda --dir files_to_replace
```

If Using System Python (Non-Anaconda)
-Run the following command:

```sh
python replace_scipy_linprog.py --env python --dir files_to_replace
```

## Installization

You can install this package from PyPI using:

`pip install si-seqfs-da`

## Reproducibility

Explore our collection of Jupyter notebooks for hands-on demonstrations of the si-seqfs-da package in action.

- Example for computing $p$-value for SeqFS after DA
This includes Forward and Backward Sequential Feature Selection (SeqFS) with optional evaluation criteria such as: AIC, BIC, Adjusted $R^2$
```
>> ex1_compute_pvalue.ipynb
```

- Check the uniformity of the $p$-values of Forward and Backward Sequential Selection under the null hypothesis $H_0$
```
>> ex2_validity_of_pvalue.ipynb
```

- Check the uniformity of the $p$-values of Forward and Backward Sequential Selection with criteria under the null hypothesis $H_0$
```
>> ex3_validity_of_pvalue_extension.ipynb
```