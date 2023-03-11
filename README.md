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
```ruby
import spaco as spaco
```
### Rank selection
ranks: a  1D array of candidate ranks for consideration, ordered from small to large

max_iter: at each rank, we maximum number iteration for SPACO estimation using all data.

cv_iter: number of iteration in cross-validation after initializing the model parameteres using the full model
```ruby
spaco.rank_selection_function(X = X, O = O, Z = Z, time_stamps = time_stamps, ranks=ranks, early_stop = True,
                            max_iter = 30, cv_iter = 5)                    
```
### Runing SPACO with given rank (=integer)
```ruby
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
```
### Randomization test from cross-fitting
Zconditional: N by q by B, Randomized auxiliary covariates using conditional randomization

Zmarginal: N by q by B, Randomized auxiliary covariates using permutation
#### (After trained the spaco model)
```ruby
train_ids, test_ids = spaco.cutfoldid(n = I, nfolds = 5, random_state = 2022)
spaco_fit.cross_validation_train( train_ids,
                               test_ids,
                               max_iter=10,
                               min_iter = 1,
                               tol=1e-3,
                               trace = True)
delta = 0;tol = 0.01;fixbeta0 = False;method = "cross";nfolds = 5;random_state = 0
feature_eval = spaco.CRtest_cross(spaco_fit, type=method, delta=delta)
```
#### precalculate quantities that stay the same for different Z
```ruby
feature_eval.precalculation()
feature_eval.cut_folds(nfolds=nfolds, random_state=random_state)
feature_eval.beta_fun_full(nfolds=nfolds, max_iter=1, tol= tol, fixbeta0=fixbeta0)
```
#### drop each feature j/beta[j] and refit beta[-j]
```ruby
for j in np.arange(feature_eval.Z.shape[1]):
    feature_eval.beta_fun_one(nfolds=nfolds, j=j, max_iter=1, fixbeta0=fixbeta0)
 ```
#### Get the refitted beta[j] using randomized quantities
```ruby 
for j in np.arange(feature_eval.Z.shape[1]):
    feature_eval.precalculation_response(j=j)
    #inplace = True: save to class object directly:  saved in   feature_eval.coef_partial, feature_eval.coef_marginal
    feature_eval.coef_partial_fun(j = j, inplace=True)
    feature_eval.coef_marginal_fun(j = j, inplace=True)
```
#### calculate the test statistics and compare them to obtain the p-values 
```ruby
feature_eval.coef_partial_random = np.zeros((feature_eval.coef_partial.shape[0],
                                             feature_eval.coef_partial.shape[1],
                                             B))                                         
feature_eval.coef_marginal_random = np.zeros((feature_eval.coef_partial.shape[0],
                                             feature_eval.coef_partial.shape[1],
                                             B))                                         
for j in np.arange(feature_eval.Z.shape[1]):
    feature_eval.coef_random_fun(Zconditional, j, type = "partial")
    feature_eval.coef_random_fun(Zmarginal, j, type="marginal")

pvals_partial_empirical, pvals_partial_fitted =  feature_eval.pvalue_calculation(type = "partial",pval_fit = True, dist_name ='nct')
pvals_marginal_empirical, pvals_marginal_fitted = feature_eval.pvalue_calculation(type = "marginal",pval_fit = True, dist_name ='nct')
```

