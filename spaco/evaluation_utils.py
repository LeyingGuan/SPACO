import numpy as np
import copy
import tensorly as tl
from tensorly.decomposition import parafac
from .spaco_module import  SPACOcv, CRtest_cross
from .helpers import  pvalue_fit

def cos_alignment(U:np.array, Uhat:np.array):
    perp = np.linalg.inv(np.matmul(np.transpose(Uhat), Uhat))
    perp = np.matmul(np.matmul(Uhat, perp), np.transpose(Uhat))
    R = U - np.matmul(perp, U)
    r2_res = (R**2).sum(axis = 0)
    r2_ori = (U**2).sum(axis = 0)
    resvar_percents = r2_res / r2_ori
    return resvar_percents


def cp_reconstruction(U: np.array, V: np.array, Phi: np.array):
    X = np.zeros((U.shape[0], Phi.shape[0], V.shape[0]))
    for j in np.arange(V.shape[0]):
        for t in np.arange(Phi.shape[0]):
            for k in np.arange(U.shape[1]):
                X[:,t,j] += U[:,k] * V[j,k] * Phi[t,k]
    return X

class metrics():
    def __init__(self,
                 U:np.array, Phi:np.array, V:np.array,
                 U0:np.array, Phi0:np.array, V0:np.array,
                 ):
        self.U = U
        self.Phi = Phi
        self.V = V
        self.U0 = U0
        self.Phi0 = Phi0
        self.V0 = V0
        self.alignment = dict()
    def component_alignment(self):
        self.alignment['U'] = cos_alignment(U = self.U0, Uhat = self.U)
        self.alignment['U'] = cos_alignment(U=self.U0,
                                            Uhat=self.U)
        self.alignment['V'] = cos_alignment(U=self.V0,
                                            Uhat=self.V)
        self.alignment['Phi'] = cos_alignment(U=self.Phi0,
                                            Uhat=self.Phi)
    def reconstruction_error(self, O: np.array):
        X = cp_reconstruction(U=self.U, V = self.V, Phi=self.Phi)
        X0 = cp_reconstruction(U=self.U0, V = self.V0, Phi=self.Phi0)
        idx_full = np.where(~np.isnan(X))
        self.alignment['reconstruct_cor_full'] = np.corrcoef(X[idx_full], X0[idx_full])[0,1]
        self.alignment['reconstruct_mse_rel_full'] = np.mean((X[idx_full]-X0[idx_full])**2)/np.mean(X0[idx_full]**2)
        idx_obs = np.where(O == 1)
        self.alignment['reconstruct_cor_obs'] = np.corrcoef(X[idx_obs], X0[idx_obs])[0, 1]
        self.alignment['reconstruct_mse_rel_obs'] = np.mean((X[idx_obs] - X0[idx_obs]) ** 2) / np.mean(X0[idx_obs] ** 2)
        idx_miss = np.where(O == 0)
        self.alignment['reconstruct_cor_miss'] = np.corrcoef(X[idx_miss], X0[idx_miss])[0, 1]
        self.alignment['reconstruct_mse_rel_miss'] = np.mean((X[idx_miss] - X0[idx_miss]) ** 2) / np.mean(X0[idx_miss] ** 2)


class comparison_pipe():
    def __init__(self, Phi0, V0, U0,
                 X, Z, O, time_stamps):
        self.Phi0 = Phi0
        self.V0 = V0
        self.U0 = U0
        self.X = X
        self.Z = Z
        self.O = O
        self.time_stamps = time_stamps
        self.signal = cp_reconstruction(U=self.U0, V = self.V0, Phi=self.Phi0)
        self.eval_dict = dict()
        self.res_cp = dict()
        self.spaco_fit = None
    def compare_run(self, rank = 3, max_iter = 30):
        #run CP
        x = self.X.copy()
        x[np.isnan(x)] = 0
        mask = self.O
        tensor_x = tl.tensor(x)
        tensor_mask = tl.tensor(mask)
        cp_result_tl = parafac(tensor_x, rank=rank,
                               mask=tensor_mask)
        cp_result_tl = cp_result_tl[1]
        self.res_cp['cp'] = dict()
        self.res_cp['cp']['U'] = cp_result_tl[0]
        self.res_cp['cp']['Phi'] = cp_result_tl[1]
        self.res_cp['cp']['V'] = cp_result_tl[1]
        self.eval_dict['cp'] = metrics(U=cp_result_tl[0],
                                  Phi=cp_result_tl[1],
                                  V=cp_result_tl[2],
                                  U0=self.U0, Phi0=self.Phi0,
                                  V0=self.V0)
        self.eval_dict['cp'].component_alignment()
        self.eval_dict['cp'].reconstruction_error(O=self.O)
        ##run SupCP random
        data_obj = dict(X=self.X, O=self.O, Z=self.Z,
                        time_stamps=self.time_stamps,
                        rank=rank)
        try:
            SupCP_random = SPACOcv(data_obj)
            SupCP_random.train_preparation(run_prepare=True,
                                           run_init=False,
                                           mean_trend_removal=False,
                                           smooth_penalty=False)
            SupCP_random.train(update_cov=True,
                        update_sigma_mu=True,
                        update_sigma_noise=True,
                        lam1_update=False, lam2_update=False,
                        max_iter=max_iter, min_iter=1,
                        tol=1e-4, trace=True
                        )
            self.res_cp['SupCP_random'] = dict()
            self.res_cp['SupCP_random'][
                'U'] = SupCP_random.mu
            self.res_cp['SupCP_random'][
                'Phi'] = SupCP_random.Phi
            self.res_cp['SupCP_random'][
                'V'] = SupCP_random.V
            self.eval_dict['SupCP_random'] = metrics(
                U=SupCP_random.mu,
                Phi=SupCP_random.Phi,
                V=SupCP_random.V,
                U0=self.U0,
                Phi0=self.Phi0,
                V0=self.V0)

            self.eval_dict[
                'SupCP_random'].component_alignment()
            self.eval_dict[
                'SupCP_random'].reconstruction_error(
                O=self.O)
        except ValueError:
            pass
        ##SupCP init_propse
        try:
            SupCP_functional = SPACOcv(data_obj)
            SupCP_functional.train_preparation(run_prepare=True,un_init=True,mean_trend_removal=False,smooth_penalty=False)
            SupCP_functional.train(update_cov=True,
                        update_sigma_mu=True,
                        update_sigma_noise=True,
                        lam1_update=False, lam2_update=False,
                        max_iter=max_iter, min_iter=1,
                        tol=1e-4, trace=True
                        )
            self.res_cp['SupCP_functional'] = dict()
            self.res_cp['SupCP_functional']['U'] =SupCP_functional.mu
            self.res_cp['SupCP_functional']['Phi'] = SupCP_functional.Phi
            self.res_cp['SupCP_functional']['V'] = SupCP_functional.V
            self.eval_dict['SupCP_functional'] = metrics(U=SupCP_functional.mu,
                                         Phi=SupCP_functional.Phi,
                                         V=SupCP_functional.V,
                                         U0=self.U0,
                                         Phi0=self.Phi0,
                                         V0=self.V0)

            self.eval_dict['SupCP_functional'].component_alignment()
            self.eval_dict['SupCP_functional'].reconstruction_error(O=self.O)
        except ValueError:
            pass
        #SPACO
        try:
            spaco= SPACOcv(data_obj)
            spaco.train_preparation(run_prepare=True,
                                    run_init=True,
                                    mean_trend_removal=False,
                                    smooth_penalty=True)
            spaco.train(update_cov=True,
                        update_sigma_mu=True,
                        update_sigma_noise=True,
                        lam1_update=True, lam2_update=True,
                        max_iter=max_iter, min_iter=1,
                        tol=1e-4, trace=True
                        )
            self.res_cp['spaco'] = dict()
            self.res_cp['spaco']['U'] =spaco.mu
            self.res_cp['spaco']['Phi'] = spaco.Phi
            self.res_cp['spaco']['V'] = spaco.V
            self.eval_dict['spaco'] = metrics(U=spaco.mu,
                                         Phi=spaco.Phi,
                                         V=spaco.V,
                                         U0=self.U0,
                                         Phi0=self.Phi0,
                                         V0=self.V0)

            self.eval_dict['spaco'].component_alignment()
            self.eval_dict['spaco'].reconstruction_error(O=self.O)
            self.spaco_fit = spaco
        except ValueError:
            pass
        #SPACO-
        try:
            data_obj['Z'] = None
            spaco_= SPACOcv(data_obj)
            spaco_.train_preparation(run_prepare=True,
                                    run_init=True,
                                    mean_trend_removal=False,
                                    smooth_penalty=True)
            spaco_.train(update_cov=True,
                        update_sigma_mu=True,
                        update_sigma_noise=True,
                        lam1_update=True, lam2_update=True,
                        max_iter=max_iter, min_iter=1,
                        tol=1e-4, trace=True
                        )
            self.res_cp['spaco_'] = dict()
            self.res_cp['spaco_']['U'] =spaco_.mu
            self.res_cp['spaco_']['Phi'] = spaco_.Phi
            self.res_cp['spaco_']['V'] = spaco_.V
            self.eval_dict['spaco_'] = metrics(U=spaco_.mu,
                                         Phi=spaco_.Phi,
                                         V=spaco_.V,
                                         U0=self.U0,
                                         Phi0=self.Phi0,
                                         V0=self.V0)

            self.eval_dict['spaco_'].component_alignment()
            self.eval_dict['spaco_'].reconstruction_error(O=self.O)
        except ValueError:
            pass
        self.eval_dict['empirical'] = dict()
        X = self.X
        X0 = self.signal
        idx_full = np.where(~np.isnan(X))
        self.eval_dict['empirical']['reconstruct_cor_full'] = np.corrcoef(X[idx_full], X0[idx_full])[0, 1]
        self.eval_dict['empirical']['reconstruct_mse_rel_full'] = np.mean((X[idx_full] - X0[idx_full]) ** 2) / np.mean(X0[idx_full] ** 2)
        idx_obs = np.where(self.O == 1)
        self.eval_dict['empirical']['reconstruct_cor_obs'] = np.corrcoef(X[idx_obs], X0[idx_obs])[0, 1]
        self.eval_dict['empirical']['reconstruct_mse_rel_obs'] = np.mean((X[idx_obs] - X0[idx_obs]) ** 2) / np.mean(X0[idx_obs] ** 2)
        idx_miss = np.where(self.O == 0)
        self.eval_dict['empirical']['reconstruct_cor_miss'] = np.corrcoef(X[idx_miss], X0[idx_miss])[0, 1]
        self.eval_dict['empirical']['reconstruct_mse_rel_miss'] = np.mean((X[idx_miss] - X0[idx_miss]) ** 2) / np.mean(X0[idx_miss] ** 2)


def CRtest_pvals(spaco, Zconditional = None, Zmarginal = None,
                 nfolds = 5, dist_name = 'nct',
                 method = 'cross', trace = False, random_state=0):
    I = spaco.Z.shape[0]; q = spaco.Z.shape[1]
    if method != 'cross':
        method = 'spaco'
    if Zconditional is not None:
        B1 = Zconditional.shape[2]
    else:
        B1 = None
    if Zmarginal is not None:
        B2 = Zmarginal.shape[2]
    else:
        B2 = None
    if Zconditional is None and Zmarginal is None:
        print('No randomized variables are provided.')
        return
    delta = 0; tol = 0.01; fixbeta0 = False;
    feature_eval = CRtest_cross(spaco, type = method, delta = delta)
    feature_eval.precalculation0()
    feature_eval.precalculation()
    feature_eval.beta_tmp = np.zeros(feature_eval.beta_tmp.shape)
    feature_eval.cut_folds(nfolds=nfolds, random_state=random_state)
    feature_eval.beta_fun_full(nfolds=nfolds, max_iter=1, tol= tol, fixbeta0=fixbeta0)
    # estimate the leave-one-feature out coefficients
    for j in np.arange(feature_eval.Z.shape[1]):
        if trace:
            print(j)
        feature_eval.beta_fun_one(nfolds=nfolds, j=j,max_iter=1,fixbeta0=fixbeta0)
    #calculate the test statistics for conditional and mariginal independence
    for j in np.arange(feature_eval.Z.shape[1]):
        feature_eval.precalculation_response(j=j)
        feature_eval.coef_partial_fun(j = j, joint=False)
        feature_eval.coef_marginal_fun(j = j, joint=False)
    #calculate the randomized test statistics
    ###null distributions
    if B1 is not None:
        coef_random_marginal = np.zeros((spaco.Z.shape[1], spaco.K, B1))
    if B2 is not None:
        coef_random_partial = np.zeros((spaco.Z.shape[1], spaco.K, B2))
    feature_eval_random = copy.deepcopy(feature_eval)
    for j in np.arange(feature_eval.Z.shape[1]):
        print(j)
        feature_eval_random.Z = feature_eval.Z.copy()
        feature_eval_random.precalculation_response(j=j)
        if B1 is not None:
            for b in np.arange(B1):
                feature_eval_random.Z[:, j] = Zconditional[:, j,b].copy()
                feature_eval_random.coef_partial_fun(j=j, joint=False)
                coef_random_partial[j, :,b] = feature_eval_random.coef_partial[j,:].copy()
        if B2 is not None:
            for b in np.arange(B2):
                feature_eval_random.Z[:, j] = Zmarginal[:, j,b].copy()
                feature_eval_random.coef_marginal_fun(j=j,joint=False)
                coef_random_marginal[j, :,b] = feature_eval_random.coef_marginal[j,:].copy()
    if B1 is not None:
        pvals_fitted_partial = np.zeros((feature_eval.Z.shape[1], spaco.K))
        pvals_empirical_partial = np.zeros((feature_eval.Z.shape[1], spaco.K))
    else:
        pvals_fitted_partial = None
        pvals_empirical_partial = None
    if B2 is not None:
        pvals_fitted_marginal = np.zeros((feature_eval.Z.shape[1], spaco.K))
        pvals_empirical_marginal = np.zeros((feature_eval.Z.shape[1], spaco.K))
    else:
        pvals_empirical_marginal = None
        pvals_fitted_marginal = None
    for j in np.arange(pvals_fitted_marginal.shape[0]):
        if trace:
            print(j)
        for k in np.arange(pvals_fitted_marginal.shape[1]):
            if B1 is not None:
                nulls = coef_random_partial[j, k, :]
                #remove infinite values
                idx = np.where(~np.isnan(nulls))[0]
                pvals_empirical_partial[j,k] = (np.sum(np.abs(nulls[idx])>=np.abs(feature_eval.coef_partial[j,k]))+1)/(len(idx)+1)
                pvals_fitted_partial[j, k] = pvalue_fit(z=feature_eval.coef_partial[j, k],nulls=nulls[idx], dist_name=dist_name)
            if B2 is not None:
                nulls = coef_random_marginal[j, k, :]
                # check if zeros variance
                idx = np.where(~np.isnan(nulls))[0]
                pvals_empirical_marginal[j,k] = (np.sum(np.abs(nulls[idx])>=np.abs(feature_eval.coef_marginal[j,k]))+1)/(len(idx)+1)
                pvals_fitted_marginal[j,k] = pvalue_fit(z = feature_eval.coef_marginal[j,k], nulls = nulls[idx], dist_name = dist_name)
    return pvals_fitted_partial, pvals_fitted_marginal, pvals_empirical_partial, pvals_empirical_marginal






