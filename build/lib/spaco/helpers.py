'''
FUNCTIONS OVERVIEW

BASIC helpers
unfold: get the modes of the X tensor and OBS tensor.

basis_creator: create basis for the penalty functions for Phi.

Uconstruct: estimate U given PhiV (perhaps regularized) with regression.

meanCurve_update,  subjectCV_mean: remove the mean time curve if needed.

ADVANCED helpers

prepare: get the input data and prepare it for ininitalization.

FPCA_smooth: functional PCA for sparse longitudinal data

initialization: initialization object for the parameters



'''
import warnings
import sys
import numpy as np
import random
import mxnet
import copy
import pandas as pd
from sklearn.model_selection import KFold
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import tensorly as tl
from tensorly.decomposition import parafac
import scipy
from rpy2.robjects import r
import torch
import os
import tensorflow as tf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
as_null = r['as.null']
pandas2ri.activate()
stats = importr('stats')
base = importr('base')
glmnet_package = importr('glmnet')
glmnet = glmnet_package.glmnet
cv_glmnet = glmnet_package.cv_glmnet

def seed_everything(seed=2021):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    mxnet.random.seed(seed)
    tf.random.set_seed(seed)

def cutfoldid(n, nfolds, random_state = 2022):
    k_fold = KFold(nfolds, shuffle=True,
                   random_state=random_state)
    subject_ids = np.arange(n)
    split_id_obj = k_fold.split(subject_ids)
    train_ids = []
    test_ids = []
    for train_index, test_index in split_id_obj:
        train_ids.append(train_index.copy())
        test_ids.append(test_index.copy())
    return train_ids, test_ids

def unfold(X, Obs):
    # I * (TJ)
    Xmode1 =  np.hstack([X[:, :, i] for i in range(X.shape[2])])
    Obs_mode1 = np.hstack([Obs[:, :, i] for i in range(X.shape[2])])
    # J * (IT)
    Xmode2 = np.hstack([np.transpose(X[:, i, :]) for i in range(X.shape[1])])
    Obs_mode2 = np.hstack([np.transpose(Obs[:, i, :]) for i in range(X.shape[1])])
    # T * (IJ)
    Xmode3 = np.hstack([np.transpose(X[:, :, i]) for i in range(X.shape[2])])
    Obs_mode3 = np.hstack([np.transpose(Obs[:, :, i]) for i in range(X.shape[2])])
    return Xmode1, Obs_mode1, Xmode2, Obs_mode2, Xmode3, Obs_mode3

def Uconstruct(X1, O1, PhiV, eps):
    I = X1.shape[0]
    K = PhiV.shape[1]
    U = np.zeros((I, K))
    Xhat = np.zeros((I, PhiV.shape[0]))
    for i in np.arange(I):
        x = X1[i, :]
        o = O1[i, :]
        o = np.where(o == 1)[0]
        z = PhiV[o, :]
        x = x[o]
        A = np.matmul(np.transpose(z),z) + eps * np.identity(K)
        a = np.matmul(np.transpose(z), x)
        U[i, :] = np.linalg.solve(A, a)
        Xhat[i, o] = np.matmul(PhiV[o, :], U[i, :])
    return U, Xhat

def basis_creator(T0, kappa = 1e-2):
    basis = np.identity(len(T0))
    left = np.zeros((len(T0)-1, len(T0)))
    for t in np.arange(len(T0)-1):
        a = 1.0/(T0[t+1] - T0[t])
        left[t, t] = a
        left[t, t + 1] = -a
    Omega = np.matmul(np.transpose(left), left)
    Omega = Omega + kappa * np.identity(len(T0))
    h = len(T0)
    return basis, Omega, h


def FPCA_smooth(Wtotal, h, num_unique_time, K):
    W_smooth = np.zeros((num_unique_time, num_unique_time))
    # off digonal estimation
    y = []
    loc1 = []
    loc2 = []
    for (s,t) in Wtotal.keys():
        m = len(Wtotal[(s,t)])
        loc1 = loc1 + [s] * m
        loc2 = loc2 + [t] * m
        y = y +list(Wtotal[(s,t)].values())
    y = np.array(y)
    loc1 = np.array(loc1)
    loc2 = np.array(loc2)
    #off diagonals
    ll = np.where(loc1 != loc2)[0]
    y0 = y[ll]
    loc10 = loc1[ll]
    loc20 = loc2[ll]
    lin_model = LinearRegression(fit_intercept=True)
    for s in np.arange(num_unique_time):
        for t in np.arange(num_unique_time):
            if s < t:
                x0 = np.transpose(np.vstack([loc10 - s, loc20 - t]))
                kers = np.exp(-(x0[:,0]**2 + x0[:,1]**2)/(2*h))
                lin_model.fit(x0, y0, kers)
                W_smooth[s,t] = lin_model.intercept_
                W_smooth[t,s] = lin_model.intercept_
    #diagonal
    ll = np.where(loc1 == loc2)[0]
    y0 = y[ll]
    loc10 = loc1[ll]
    loc20 = loc2[ll]
    for s in np.arange(num_unique_time):
        x0 = loc10 - s
        kers = np.exp(-(x0 ** 2) / (2 * h))
        lin_model.fit(x0.reshape((len(x0),1)), y0, kers)
        W_smooth[s, s] = lin_model.intercept_
    svd0 = np.linalg.svd(W_smooth)
    B = svd0[0][:,:K]
    diags = svd0[1]
    return W_smooth, B, diags


def penalty_search(A, Omega, df0=10, tol=1e-4, \
                   max_iter=100):
    it = 0;
    err = tol * 2
    tmp1 = np.linalg.svd(A)[1]
    tmp2 = np.linalg.svd(Omega)[1]
    cur_max = np.max(tmp1) / (np.min(tmp2)+1e-6)
    cur_min = 1e-10
    while it < max_iter and np.abs(err) > tol:
        it = it + 1
        middle = (cur_min + cur_max) / 2.0
        Ainv = np.linalg.inv(A + middle * Omega)
        df = np.sum(np.diag(np.matmul(Ainv, A)))
        err = df - df0
        if err > tol:
            cur_min = middle
        elif err < -tol:
            cur_max = middle
    return df, middle


def meanCurve_update(Psi, X, Omega, obs0, lam0):
    B0 = np.zeros((X.shape[1], X.shape[2]))
    A0 = np.matmul(np.transpose(Psi),Psi)
    X0 = np.zeros((int(np.sum(obs0)), X.shape[2]))
    for j in np.arange(X.shape[2]):
        x = X[:,:,j]
        x = x[obs0 == 1.0]
        X0[:,j] = x.copy()
        A = A0 + lam0 * Omega
        B = np.matmul(np.transpose(Psi), x)
        B0[:, j] = np.matmul(np.linalg.inv(A), B)
    return B0, X0


def subjectCV_mean(X, Psi, obs0, s0, Omega, lams):
    #get the overall estimate
    nlam = len(lams)
    cv = np.zeros((X.shape[0], X.shape[2], nlam))
    for k in np.arange(nlam):
        B0, X0 = meanCurve_update(Psi, X, Omega, obs0, lams[k])
        R = X0 - np.matmul(Psi, B0)
        A0 = np.matmul(np.transpose(Psi), Psi) + lams[k] * Omega
        A0 = np.linalg.inv(A0)
        for j in np.arange(X.shape[2]):
            rj = R[:,j]
            for i in np.arange(X.shape[0]):
                si = np.where(s0 == i)[0]
                psii = Psi[si,:]
                Vi = np.identity(len(si)) - np.matmul(np.matmul(psii,A0), np.transpose(psii))
                Vi = np.linalg.inv(Vi)
                rij = rj[si]
                cv[i,j,k] = 1.0/len(si) * np.sum(np.matmul(Vi, rij)**2)
    cv = np.mean(np.mean(cv, axis = 0),axis = 0)
    lam0 = lams[np.argmin(cv)]
    B0, X0 = meanCurve_update(Psi, X, Omega, obs0, lam0)
    return cv, B0

def prepare(X,  Obs, T0, mean_removel = True, nlam = None, lams = None,  kappa = 1e-2):
    basis, Omega, h = basis_creator(T0, kappa=kappa)
    R = X.copy()
    B0 = None; cv = None; dfs = None
    if mean_removel:
        if nlam is None and lams is None:
            print("no penalty(s) supplied!")
            return
        T = np.zeros((X.shape[0],X.shape[1]), dtype = int)
        S = np.zeros((X.shape[0],X.shape[1]), dtype = int)
        for i in np.arange(X.shape[0]):
            T[i,:] = np.array(np.arange(len(T0)),dtype = int)
            S[i,:] = i
        obs0 = Obs[:,:,0]
        ts = T[obs0 == 1]
        s0 = S[obs0 == 1]
        Psi = basis[ts,:]
        A = np.matmul(np.transpose(Psi), Psi)
        if lams is None:
            #penalty search
            dfs = 1.0+(1.0-np.arange(nlam)/(nlam-1.0)) * ((len(ts)/np.log(len(ts)))-1.0)
            lams = np.zeros((nlam))
            for l in np.arange(nlam):
                df_hat, lams[l]= penalty_search(A, Omega, df0=dfs[l], tol= 1e-4,max_iter=1000)
        else:
            nlam = len(lams)
        cv, B0 = subjectCV_mean(X, Psi, obs0, s0, Omega, lams)
        tmp = np.matmul(basis, B0)
        for j in np.arange(X.shape[2]):
            for i in np.arange(X.shape[0]):
                R[i,:,j] = R[i,:,j] - tmp[:,j]
    return Omega, h, R, B0, cv, lams, dfs

def seed_everything(seed=2021):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    mxnet.random.seed(seed)
    tf.random.set_seed(seed)


'''
Check compatability
'''
def my_glmnet(x, y, lam, nlam, penalty_factor):
    fitted = glmnet(x=x, y= y.reshape(len(y),1), family = "gaussian",
                    intercept=False, nlambda=nlam, penalty_factor=penalty_factor)
    coefficients0 = stats.coef(fitted, s=lam)
    coefficients0 = robjects.r['summary'](coefficients0)
    coefficients0 = np.array(coefficients0)
    coefficients = np.zeros((x.shape[1]))
    for j in np.arange(coefficients0.shape[0]):
        coefficients[int(coefficients0[j, 0]) - 2] = coefficients0[j, 2]
    return coefficients


def my_cv_glmnet(x, y,penalty_factor,  nlam, nfolds, foldid = None):
    if foldid is None:
        foldid = as_null()
    fitted = cv_glmnet(x=x, y= y.reshape(len(y),1), intercept=False,
                       nfolds=nfolds, foldid = foldid, nlambda=nlam,
                       penalty_factor=penalty_factor)
    coefficients0 = stats.coef(fitted, s="lambda.min")
    extracted = tuple(fitted)
    lambdas = extracted[0]
    cvm = extracted[1]
    cvsd = extracted[2]
    coefficients0 = robjects.r['summary'](coefficients0)
    coefficients0 = np.array(coefficients0)
    coefficients = np.zeros((x.shape[1]))
    idx0 = np.argmin(cvm)
    lambda1min = lambdas[idx0]
    for j in np.arange(coefficients0.shape[0]):
        coefficients[int(coefficients0[j, 0]) - 2] = coefficients0[j, 2]
    return coefficients, lambda1min, cvm, cvsd, lambdas

def beta_fit(Z, transform_mat, transform_vec, beta,intercepts,lambda2, fit_intercept = True,
             lam2_update = True, nlam2 = 100, nfolds = 5, foldid = as_null(),max_iter = 1, tol = 0.01):
    augZ = Z.copy()
    err = 2.0 * tol
    it = 0
    beta_prev = beta.copy()
    intercepts_prev = intercepts.copy()
    K = transform_vec.shape[1]
    while it < max_iter and err > tol:
        if fit_intercept is True:
            augZ = np.hstack([Z, np.ones((Z.shape[0], 1), dtype=float)])
        for k in np.arange(K):
            mu_trans = np.zeros(Z.shape[0])
            Z_trans = np.zeros(augZ.shape)
            bb = np.delete(beta_prev, k, axis=1)
            fit_others = np.matmul(Z, bb)
            if fit_intercept:
                fit_others = fit_others + np.delete(intercepts_prev, k)
            #create the transformed z and response
            for i in np.arange(Z.shape[0]):
                tmp = transform_mat[i, k, :]
                tmp = np.delete(tmp, k)
                mu_trans[i] = (transform_vec[i, k] - np.sum(fit_others[i, :] * tmp))/np.sqrt(transform_mat[i, k, k])
                Z_trans[i, :] = augZ[i, :] * np.sqrt(transform_mat[i, k, k])
            penalty_factor = np.ones(Z_trans.shape[1])
            if fit_intercept:
                penalty_factor[Z_trans.shape[1] - 1] = 0.0
            if lam2_update:
                coefficients, lambda1min, cvm, cvsd, lambdas = my_cv_glmnet(x=Z_trans, y=mu_trans,
                             penalty_factor=penalty_factor, nlam= nlam2, nfolds=nfolds, foldid = foldid)
                lambda2[k] = lambda1min
            else:
                coefficients = my_glmnet(x=Z_trans, y=mu_trans, lam=lambda2[k], nlam=100, penalty_factor=penalty_factor)
            p0 = len(coefficients)
            if fit_intercept:
                intercepts[k] = coefficients[p0 - 1]
                p0 = p0 - 1
            beta[:, k] = coefficients[:p0].copy()
        err = np.sum((beta - beta_prev)**2)
        it += 1
        beta_prev = beta.copy()
        intercepts_prev = intercepts.copy()
    return beta, intercepts, lambda2


def binary_search_vector(a: np.array, b:np.array, norm_constraint = 1.0):
    if np.min(a) <= 0:
        sys.exit('encountered nonPD input during the subroutine updating V.')
    z = b/a
    b2 = np.sum(b*b)
    norm2_0 = np.sqrt(np.sum(z**2))
    if norm2_0 > norm_constraint + 1e-8:
        lam_min = 0.0
        lam_max = np.sqrt(b2)/norm_constraint - np.min(a)
    elif norm2_0 < norm_constraint - 1e-8:
        idx = np.argmin(a)
        lam_min = -(a[idx] - np.abs(b[idx])/norm_constraint)
        lam_max = 0.0
    else:
        lam_min = 0.0
        lam_max = 0.0
    lam_cur = 0.0
    while np.abs(norm2_0 - norm_constraint) > 1e-8:
        if norm2_0 >norm_constraint:
            lam_min = lam_cur
            lam_cur = (lam_cur + lam_max)/2.0
        else:
            lam_max = lam_cur
            lam_cur = (lam_cur + lam_min) / 2.0
        z = b / (a + lam_cur)
        norm2_0 = np.sqrt(np.sum(z**2))
    return lam_cur


def phi_solver(a, b, Omega, lam1, ridge_traj, h):
    A = Omega * lam1 + np.diag(a + ridge_traj)
    Asvd = np.linalg.svd(A)
    b1 = np.matmul(Asvd[2], b)
    a1 = Asvd[1]
    laplace_lam = binary_search_vector(a1, b1, norm_constraint=h)
    phinew = b1 / (a1 + laplace_lam)
    phinew = np.matmul(Asvd[0], phinew)
    return phinew


def cross_posterior(xte, ote, zte, s2, sigma_mu, PhiV,  beta = None, intercepts = None, fit_intercept = True):
    K = PhiV.shape[1]
    I = xte.shape[0]
    mu  = np.zeros((I, K))
    cov = np.zeros((I, K, K))
    transform_mat = np.zeros((I, K, K))
    transform_vec =np.zeros((I, K))
    temp_vec = np.zeros((I, K))
    for i in np.arange(I):
        xte0 = xte[i, :]
        obs0 = ote[i,:]
        obs0 = np.where(obs0 == 1)[0]
        PhiV0 = PhiV[obs0, :]
        xte0 = xte0[obs0]
        s20 = s2[obs0]
        tmp1 = np.matmul(np.transpose(PhiV0) / s20, PhiV0)
        tmp2 = np.matmul(np.transpose(PhiV0), xte0 / s20)
        tmp3 = np.zeros((K))
        M = np.diag(1.0 / sigma_mu)
        tmp4 = tmp1 + M
        tmp4 = np.linalg.inv(tmp4)
        cov[i, :, :] = tmp4.copy()
        M = M - np.matmul(np.matmul(M, cov[i, :, :]),M)
        if zte is not None:
            tmp3 = np.matmul(zte[i,],beta) / sigma_mu
            if fit_intercept:
                tmp3 = tmp3 + intercepts / sigma_mu
        mu[i, :] = np.matmul(cov[i, :, :], tmp2 + tmp3)
        transform_mat[i, :, :] = M.copy()
        transform_vec[i, :] = np.matmul(np.matmul(tmp2, cov[i, :, :]),np.diag(1.0 / sigma_mu))
        temp_vec[i,:] = tmp2.copy()
    return mu, cov,  transform_vec, transform_mat, temp_vec


def permutation_align_greedy(cross_fit_obj, cor_mat):
    orders  = np.zeros(cross_fit_obj.K, dtype = int)
    columns = np.arange(cross_fit_obj.K)
    rows = np.arange(cross_fit_obj.K)
    abs_cor_mat = np.abs(cor_mat)
    idx_final = np.zeros(cross_fit_obj.K,dtype = int)
    for it in np.arange(cross_fit_obj.K):
        s1 = np.zeros(abs_cor_mat.shape[0])
        s2 = np.zeros(abs_cor_mat.shape[0])
        idx = np.zeros(abs_cor_mat.shape[0], dtype=int)
        for k in np.arange(abs_cor_mat.shape[1]):
            idx[k] = np.argmax(abs_cor_mat[:,k])
            s1[k] = abs_cor_mat[idx[k],k]
            if len(idx)>1:
                s2[k] = np.max(np.delete(abs_cor_mat[:,k],idx[k]))
            else:
                s2[k] = 0.0
        max_id = np.argmax(s1-s2)
        orders[columns[max_id]] = rows[idx[max_id]].copy()
        columns = np.delete(columns, max_id)
        rows = np.delete(rows, idx[max_id])
        abs_cor_mat = np.delete(abs_cor_mat, max_id, axis = 1)
        abs_cor_mat = np.delete(abs_cor_mat, idx[max_id],axis=0)
    signs = np.zeros(cross_fit_obj.K)
    for k in np.arange(cross_fit_obj.K):
        signs[k] = 1
        if cor_mat[orders[k], k] < 0:
            signs[k] = -1
    cross_fit_obj.V = cross_fit_obj.V[:,orders] * signs
    cross_fit_obj.Phi = cross_fit_obj.Phi[:,orders]
    cross_fit_obj.sigmaF = cross_fit_obj.sigmaF[orders]
    if cross_fit_obj.beta is not None:
        cross_fit_obj.beta =cross_fit_obj.beta[:,orders] * signs
    if cross_fit_obj.intercepts is not None:
        cross_fit_obj.intercepts =  cross_fit_obj.intercepts[orders]*signs
    if cross_fit_obj.lambda2 is not None:
        cross_fit_obj.lambda2 =cross_fit_obj.lambda2[orders]
    cross_fit_obj.lambda1 = cross_fit_obj.lambda1[orders]
    return cross_fit_obj




def beta_fit_cleanup(Z, beta, intercepts, delta, prevec, cov, noiseCov,
                     test_ids, sigmaF, lambda2, factor_idx = None,
                     fit_intercept = True, max_iter = 1, tol  = 0.01,
                     lam2_update = True,  nlam2 = 100, nfolds = 5, foldid= None,
                     beta_fix = None, intercepts_fix = None, iffix = False):
    ##create y and z
    I = Z.shape[0]
    K = beta.shape[1]
    augZ = Z.copy()
    err = 2.0 * tol
    it = 0
    beta_prev = beta.copy()
    intercepts_prev = intercepts.copy()
    if factor_idx is None:
        factor_idx = np.arange(K)
    while it < max_iter and err > tol:
        if fit_intercept is True:
            augZ = np.hstack([Z, np.ones((Z.shape[0], 1), dtype=float)])
        for k in factor_idx:
            mu_trans = np.zeros(Z.shape[0])
            Z_trans = np.zeros(augZ.shape)
            if iffix is False:
                bb = np.delete(beta_prev, k, axis=1)
            else:
                bb = np.delete(beta_fix, k, axis=1)
            tmp0 = np.matmul(Z, bb)
            if fit_intercept:
                if iffix is False:
                    tmp0 = tmp0 + np.delete(intercepts_prev, k)
                else:
                    tmp0 = tmp0 + np.delete(intercepts_fix,k)
            for fold_id in np.arange(len(test_ids)):
                for i0 in np.arange(len(test_ids[fold_id])):
                    i = test_ids[fold_id][i0]
                    wik = 1.0/np.sqrt(noiseCov[i,k,k])
                    mu_trans[i] = np.sum(cov[i,k,:]*prevec[i,:])
                    for l in np.arange(K):
                        if l != k:
                            mu_trans[i] += delta * cov[i,k,l] * sigmaF[l,fold_id]
                    mu_trans[i] = mu_trans[i] * wik
                    Z_trans[i,:] = augZ[i,:] * (1.0 - delta * cov[i,k,k]/sigmaF[k,fold_id]) * wik
            penalty_factor = np.ones(Z_trans.shape[1])
            if fit_intercept:
                penalty_factor[Z_trans.shape[1] - 1] = 0.0
            if lam2_update:
                coefficients, lambda1min, cvm, cvsd, lambdas = my_cv_glmnet(x=Z_trans, y=mu_trans,
                             penalty_factor=penalty_factor, nlam= nlam2, nfolds=nfolds, foldid = foldid)
                lambda2[k] = lambda1min
            else:
                coefficients = my_glmnet(x=Z_trans, y=mu_trans, lam=lambda2[k], nlam=100, penalty_factor=penalty_factor)
            p0 = len(coefficients)
            if fit_intercept:
                intercepts[k] = coefficients[p0 - 1]
                p0 = p0 - 1
            beta[:, k] = coefficients[:p0].copy()
        err = np.sum((beta - beta_prev)**2)
        it += 1
        beta_prev = beta.copy()
        intercepts_prev = intercepts.copy()
    return beta, intercepts, lambda2

def pvalue_fit(z, nulls, dist_name):
    dist = getattr(scipy.stats, dist_name)
    mu = np.mean(nulls)
    s = np.std(nulls)
    t_null = (nulls - mu) / s
    t0 = np.abs((z - mu) / s)
    params = dist.fit(t_null)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    pval = dist.cdf(-t0, *arg, loc=loc, scale=scale)+1.0 - dist.cdf(t0, *arg, loc=loc, scale=scale)
    return pval
