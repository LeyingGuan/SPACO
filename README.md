# SPACO (Python Implementation of SPACO)


## Overview:

python package for spaco and associated data used in the paper.

The directory spaco contains code for the package.

The directories simulations, realdata contain code for reproducing the simulation and real data examples in the paper.

Reference: Guan, Leying. "A smoothed and probabilistic PARAFAC model with covariates." arXiv preprint arXiv:2104.05184 (2021).

## Installation.

pip install git+https://github.com/LeyingGuan/SPACO.git#egg=spaco

## Tutorial
### Input data explanations
X: a N by T by J tensor (subject by time by feature), missing value as np.nan

O: an indicator a N by T by J tensor, 1=observed, 0 = missing.

Z: auxiliary n by p covariate. Z = None = SPACO-.

time_stamps: length T vectors indicating time. It is used for creating default regularization matrix matrix.

### Load package
import spaco as spaco

### Runing SPACO with given rank (=integer)
data_obj = dict(X=X, O=O, Z=Z,time_stamps=time_stamps, rank=rank)
spaco_fit = spaco.SPACOcv(data_obj)
spaco_fit.train_preparation(run_prepare=True,
                        run_init=True,
                        mean_trend_removal=False,
                        smooth_penalty=True)

spaco_fit.train(update_cov=True,
            update_sigma_mu=True,
            update_sigma_noise=True,
            lam1_update=True, lam2_update=True,
            max_iter=30, min_iter=1,
            tol=1e-4, trace=True
            )
### Rank selection
spaco.rank_selection_function(X = X, O = O, Z = Z, time_stamps = time_stamps, ranks=ranks, early_stop = True,
                            max_iter = 30, cv_iter = 5)
                            
ranks: a  1D array of candidate ranks for consideration, ordered from small to large

max_iter: at each rank, we maximum number iteration for SPACO estimation using all data.

cv_iter: number of iteration in cross-validation after initializing the model parameteres using the full model

### Randomization test from cross-fitting
