'''
FUNCTIONS OVERVIEW
my_glmnet:
my_glmnet_cv:
beta_fit:


CLASSES OVERVIEW

initialization: PARAMETER INITIALIZATIONS
spaco: spaco with outomatic penalty selection.
spaco_cf: spaco with cross-fit.
spaco_inf: post selection inference for the spaco or spaco_cf objs.
'''
from rpy2.robjects import pandas2ri
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from tensorly.decomposition import parafac

from .helpers import *

as_null = r['as.null']
pandas2ri.activate()
stats = importr('stats')
base = importr('base')
glmnet_package = importr('glmnet')
glmnet = glmnet_package.glmnet
cv_glmnet = glmnet_package.cv_glmnet


class initialization():
    def __init__(self, X, OBS, T0, Xmodes = None, OBSmodes = None, K = 1, mod = "pca", homoNoise = True,
                   vNorm = 1, phiNorm = 1, jen_num = 10, random_num = 10, h = None, eps = 1e-10,
                 random_state = 0):
        self.X = X
        self.O = OBS
        self.T0 = T0
        self.K = K
        self.mod = mod
        self.homoNoise = homoNoise
        self.vNorm = vNorm
        self.phiNorm = phiNorm
        self.random_num = random_num
        self.jen_num = jen_num
        self.Xmodes = Xmodes
        self.Omodes = OBSmodes
        self.sigma2 = np.ones(X.shape[2])
        self.s2 = np.ones(self.K)
        self.random_state = random_state
        if self.Xmodes is None or self.Xmodes is None:
            self.Xmodes = {}
            self.Omodes = {}
            self.Xmodes['X1'],self.Omodes['O1'], self.Xmodes['X2'],self.Omodes['O2'], \
            self.Xmodes['X3'],self.Omodes['O3'] = unfold(self.X, self.O)
        self.V0 = None
        self.Phi0 = None
        self.U0 = None
        self.G = None
        self.V = None
        self.Phi = None
        self.U = None
        self.h = None
        self.eps = eps
    def Vinit(self):
        idx = np.sum(1.0 - self.Omodes['O2'], axis=0)
        idx = np.where(idx == 0)[0]
        np.random.seed(self.random_state)
        KJ = self.K
        if(KJ > self.X.shape[2]):
            KJ = self.X.shape[2]
        if self.mod == "pca":
            tmp = self.Xmodes['X2'][:, idx]
            means = np.mean(tmp, axis=0)
            tmp = tmp - means
            tmp = np.linalg.svd(tmp)
            self.V0 = tmp[0][:, :KJ]
        else:
            self.V0 = np.random.normal(shape=(self.Xmodes['X2'].shape[0], KJ))
            for k in np.arange(KJ):
                self.V0[:, k] = self.V0[:, k] / np.sqrt(np.sum(self.V0[:, k] ** 2))
    def Phiinit(self):
        num_subject, num_unique_time, num_feature = self.X.shape
        np.random.seed(self.random_state)
        KT = self.K
        if(KT > num_unique_time):
            KT = num_unique_time
        if self.mod == "pca":
            W = np.zeros((num_subject, num_unique_time, KT))
            W[:] = np.nan
            for k in np.arange(KT):
                for t in np.arange(num_unique_time):
                    W[:, t, k] = np.matmul(self.X[:, t, :], self.V0[:, k])
            Wtotal = {}
            for s in np.arange(num_unique_time):
                for t in np.arange(num_unique_time):
                    if s <= t:
                        Wtotal[(s, t)] = {}
                        for i in np.arange(self.X.shape[0]):
                            if self.O[i, t, 0] * self.O[i, s, 0] == 1:
                                Wtotal[(s, t)][i] = np.sum(W[i, t, :] * W[i, s, :])
            for s in np.arange(num_unique_time):
                for t in np.arange(num_unique_time):
                    if s <= t:
                        if (s, t) in Wtotal.keys():
                            if len(Wtotal[(s, t)]) == 0:
                                Wtotal.pop((s, t))
            if self.h is None:
                self.h = num_unique_time / np.sqrt(len(Wtotal))
            W_smooth, self.Phi0, diags = FPCA_smooth(Wtotal, h=self.h,num_unique_time=num_unique_time, K=KT)
        else:
            self.Phi0 = np.random.normal((num_unique_time, KT))
            for k in np.arange(KT):
                self.Phi0[:, k] = self.Phi0[:, k] / np.sqrt(np.sum(self.Phi0[:, k] ** 2))
    def UGinit(self):
        #column order V1Phi1, V1Phi2,...,V1PhiK,..., VKPhi1,...VKPhiK
        self.Utmp = np.zeros((self.X.shape[0], self.Phi0.shape[1] * self.V0.shape[1]))
        self.G = np.zeros((self.K, self.Phi0.shape[1], self.V0.shape[1]))
        PhiV0 = np.kron(self.V0, self.Phi0)
        for i in np.arange(self.X.shape[0]):
            x0 = self.Xmodes['X1'][i, :]
            o0 = self.Omodes['O1'][i, :]
            o0 = np.where(o0 == 1)[0]
            PhiV01 = PhiV0[o0, :]
            x0 = x0[o0]
            A = np.matmul(np.transpose(PhiV01),PhiV01) + self.eps * np.identity(PhiV0.shape[1])
            a = np.matmul(np.transpose(PhiV01), x0)
            self.Utmp[i,:] = np.linalg.solve(A, a)
        np.random.seed(self.random_state)
        tmp = np.linalg.svd(self.Utmp)
        self.U0 = tmp[0][:,:self.K]
        # tmp: K * (K^2)
        tmp = np.matmul(np.transpose(self.U0), self.Utmp)
        for k in np.arange(self.K):
            for l in np.arange(self.V0.shape[1]):
                ll = np.arange((l*self.Phi0.shape[1]),(l+1)*self.Phi0.shape[1])
                self.G[k,:,l] = tmp[k,ll]
    def Gparafac(self):
        J = self.V0.shape[0]
        T = self.Phi0.shape[0]
        I = self.U0.shape[0]
        V = np.zeros((J, self.K))
        Phi = np.zeros((T, self.K))
        PhiV = np.zeros((J*T, self.K))
        Xhat = np.zeros((I, J*T))
        errorsG = np.zeros((2))
        np.random.seed(self.random_state)
        parafacG_svd = parafac(tensor= self.G, rank = self.K, n_iter_max=10000,
                                                init='svd', svd='numpy_svd',random_state=self.random_state)[1]
        ###calculate the reconstruction error
        for k in np.arange(self.K):
            V[:,k] = np.matmul(self.V0, parafacG_svd[2][:,k])
            Phi[:, k] = np.matmul(self.Phi0,parafacG_svd[1][:,k])
            PhiV[:,k] = np.kron(V[:, k].reshape((J, 1)),Phi[:, k].reshape((T, 1))).reshape(-1)
        U, Xhat = Uconstruct(X1 = self.Xmodes['X1'], O1 = self.Omodes['O1'], PhiV = PhiV, eps = self.eps)
        #Ghat = tl.cp_to_tensor(parafacG_svd)
        errorsG[0] = np.mean((self.Xmodes['X1'][self.Omodes['O1']==1]-Xhat[self.Omodes['O1']==1])**2)
        #random_num Random
        if self.random_num > 0:
            err_tmp = np.zeros(self.random_num)
            np.random.seed(self.random_state)
            seeds_rd = np.zeros(self.random_num, dtype = int)
            for l in np.arange(self.random_num):
                seeds_rd[l] = int(self.random_state+self.random_num)
            parafacG_rd_list = list()
            for l in np.arange(self.random_num):
                seed_everything(int(seeds_rd[l]))
                parafacG_rd = parafac(tensor= self.G, rank = self.K, n_iter_max=100,
                                  init='random', random_state =int(seeds_rd[l]))[1]
                ###calculate the reconstruction error
                for k in np.arange(self.K):
                    V[:, k] = np.matmul(self.V0,parafacG_rd[2][:, k])
                    Phi[:, k] = np.matmul(self.Phi0,parafacG_rd[1][:, k])
                    PhiV[:, k] = np.kron(V[:, k].reshape((J, 1)),Phi[:, k].reshape((T, 1))).reshape(-1)
                U, Xhat = Uconstruct(X1=self.Xmodes['X1'],O1=self.Omodes['O1'], PhiV=PhiV, eps=self.eps)
                err_tmp[l] = np.mean((self.Xmodes['X1'][self.Omodes['O1']==1]-Xhat[self.Omodes['O1']==1])**2)
                parafacG_rd_list.append(copy.deepcopy(parafacG_rd))
            errorsG[1] = np.min(err_tmp)
            parafacG_rd = parafacG_rd_list[np.argmin(err_tmp)]
        else:
            errorsG[1] = np.inf
            parafacG_rd = parafacG_svd
        # jen_num Jenrich
        if errorsG[1] < errorsG[0]:
            parafacG = parafacG_rd
        else:
            parafacG = parafacG_svd
        self.A =  parafacG[0]
        self.B = parafacG[1]
        self.C = parafacG[2]
    def transform(self):
        J = self.V0.shape[0]
        T = self.Phi0.shape[0]
        I = self.U0.shape[0]
        self.PhiV = np.zeros((J * T, self.K))
        self.U = np.matmul(self.U0, self.A)
        self.Phi = np.matmul(self.Phi0, self.B)
        self.V = np.matmul(self.V0, self.C)
        for k in np.arange(self.K):
            a1 = np.sqrt(np.sum(self.Phi[:,k]**2))/self.phiNorm
            a2 = np.sqrt(np.sum(self.V[:, k] ** 2)) / self.vNorm
            self.Phi[:,k] = self.Phi[:,k]/a1
            self.V[:,k] = self.V[:,k]/a2
            self.PhiV[:, k] = np.kron(self.V[:, k].reshape((J, 1)),self.Phi[:, k].reshape((T, 1))).reshape(-1)
        U, Xhat = Uconstruct(X1=self.Xmodes['X1'],O1=self.Omodes['O1'],PhiV=self.PhiV, eps=self.eps)
        self.U = U.copy()
        self.X1hat = Xhat.copy()
    def varianceInit(self):
        self.s2 = np.mean(self.U**2, axis = 0)
        if self.homoNoise:
            self.sigma2[:] = np.sum((self.X1hat[self.Omodes['O1']==1] - self.Xmodes['X1'][self.Omodes['O1']==1])**2)/np.sum(self.Omodes['O1'])
    def run(self):
        self.Vinit()
        self.Phiinit()
        self.UGinit()
        self.Gparafac()
        self.transform()
        self.varianceInit()

'''
SPACO (Smoothed Probablistic PARAFAC with covariates)

In many clinical study, we can have subjects measured at different time points
and are interested in identifying patient sub-clusters and covariates that are
important in the clustering step.

:argument
@ X: I * T * J array time series array.
@ O: I * T * J indicator array of whether an observation is available.
@ Z: I * q covariate matrix
@ K: rank
@ Omega: M * M smooth penalty matrix
@ H: M * M norm constraint matrix.
@ Phi: T * K smooth principal components.
@ V: J * K loadings for multivariate features.
@ mu: I * K posterior means of the U matrix
@ cov: I * K * K posterior covariance of the U matrix.
@ beta: q * K coefficient matrix.
@ sigmaF2: length-K vectors, variances for uk.
@ sigma2: length-p vectors, variances for Xj.
@ lambda1: lengthe K vector, smooth penalty.
@ lambda2: length K vector, sparsity penalty.
'''
class SPACObase():
    def __init__(self, data_obj):
        self.X0 = data_obj['X']
        self.X = data_obj['X']
        self.O = data_obj['O']
        self.Z = data_obj['Z']
        self.K = data_obj['rank']
        self.time_stamps = data_obj['time_stamps']
        self.intermediantes = {}
        self.num_subjects = self.X.shape[0]
        self.num_times = self.X.shape[1]
        self.num_features =self.X.shape[2]
        self.Omega = None
        self.h = None
        self.Phi = None
        self.V = None
        self.beta = None
        self.mu = None
        self.cov = None
        self.intercepts = None
        self.sigma_mu = None
        self.sigma_noise = None
        self.smooth_penalty = True
        self.lasso_penalty = True
        self.fit_intercept = True
        self.lam1_update = True
        self.lam2_update = True
        self.nlam1 = None
        self.nlam2 = None
        self.lam1_dfmin = None
        self.lam1_dfmax = None
        self.intermediantes['lambda1s'] = None
        self.intermediantes['lambda2s'] = None
        self.intermediantes['selected'] = None
        self.intermediantes['Phis'] =None
        self.lam1criterion = None
        self.lam2criterion = None
        self.lambda1 = None
        self.lambda2 = None
        self.ridge_traj = 1e-4
        self.beta_foldid = None
        self.transform_mat= None
        self.homoNoise = True
        self.lasso_maxit = 1e4
        self.lasso_tol = 1e-8
        self.orthogonal = 0.0
        self.transform_mat = np.zeros((self.num_subjects, self.K, self.K))
        self.transform_vec = np.zeros((self.num_subjects, self.K))
        self.beta_folds = 10
    def check_init(self):
        if self.h is None:
            self.h = self.num_times
        if self.Phi is None:
            self.Phi = np.random.normal(
                size=(self.num_times, self.K))
        for k in np.arange(self.K):
            l2norm = np.sqrt(np.sum(self.Phi[:, k] ** 2) / self.h)
            self.Phi[:, k] = self.Phi[:, k] / l2norm
        if self.V is None:
            self.V = np.random.normal(size=(self.num_features, self.K))
        for k in np.arange(self.K):
            l2norm = np.sqrt(np.sum(self.V[:, k] ** 2))
            self.V[:, k] = self.V[:, k] / l2norm
        if self.beta is None:
            if self.Z is not None:
                self.beta = np.zeros(
                    (self.Z.shape[1], self.K))
        if self.intercepts is None:
            self.intercepts = np.zeros((self.K))
        if self.mu is None:
            self.mu = np.zeros((self.num_subjects, self.K))
        if self.cov is None:
            self.cov = np.zeros(
                (self.num_subjects, self.K, self.K))
        if self.sigma_mu is None:
            self.sigma_mu = np.ones((self.K)) * 10 ** 8
        # no smoothness penalty
        if self.lambda1 is None:
            self.lambda1 = np.ones((self.K))*1e-2
        # no lasso penalty
        if self.lambda2 is None:
            self.lambda2 = np.ones((self.K))*1e-2
        if self.sigma_noise is None:
            self.sigma_noise = np.ones((self.num_features))
        if self.orthogonal is None:
            self.orthogonal = 2 * np.sqrt(
                self.num_subjects) * np.log(
                self.num_subjects * self.num_times * self.num_subjects)
    def train_preparation(self, K = None,
              run_prepare = True, run_init=True,
              mean_trend_removal: bool = True,
              smooth_penalty: bool = True,
              lasso_penalty: bool = True,
              Omega= None, h = None,
              nlam1 = 10, nlam2 = 100, nlam0 = 10, kappa = 1e-2,
              Phi=None, V = None, mu = None, cov = None,
              intercepts = None, fit_intercept = True, beta = None,
              sigma_mu = None, sigma_noise = None,
              lambda1 = None, lambda2 = None,
              lam1_dfmin = None, lam1_dfmax=None,
              lam1criterion = 1.0, lam2criterion = 1.0,
              init_repeats = 10, random_state = 0,
              lasso_maxit = 1e4, lasso_tol=1e-8,
              intercept_scale = 1e2, beta_folds = 10):
        if K is not None:
            self.K = K
            self.transform_mat = np.zeros(
                (self.num_subjects, self.K, self.K))
            self.transform_vec = np.zeros(
                (self.num_subjects, self.K))
        self.Omega = Omega
        self.h = h
        self.Phi = Phi
        self.V = V
        self.mu = mu
        self.cov = cov
        self.sigma_mu = sigma_mu
        self.sigma_noise = sigma_noise
        self.fit_intercept = fit_intercept
        self.intercepts = intercepts
        self.beta = beta
        self.smooth_penalty = smooth_penalty
        self.lasso_penalty = lasso_penalty
        self.nlam1 = nlam1
        self.nlam2 = nlam2
        self.Omega = Omega
        self.h = h
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lam1_dfmin = lam1_dfmin
        self.lam1_dfmax = lam1_dfmax
        self.lam1criterion =lam1criterion
        self.lam1criterion = lam2criterion
        self.lasso_maxit = lasso_maxit
        self.lasso_tol = lasso_tol
        self.beta_folds  = beta_folds
        if run_prepare:
            print("start data preparation:")
            Omega, h, R, B0, cv, lams, dfs = \
                prepare(self.X, self.O, self.time_stamps, mean_removel=mean_trend_removal,
                        nlam=nlam0,  kappa=kappa)
            print("data preparation done.")
            self.X = R
            if self.Omega is None:
                self.Omega = Omega
            if self.h is None:
                self.h = h
        self.intermediantes['Xmod1'], self.intermediantes['O1'],\
        self.intermediantes['Xmod2'], self.intermediantes['O2'], \
        self.intermediantes['Xmod3'], self.intermediantes['O3']\
        = unfold(self.X, self.O)
        if run_init:
            ###initialization
            print("start initialization:")
            initializer = initialization(X=R, OBS=self.O,
                                         T0=self.time_stamps,
                                         K=self.K,
                                         eps=1.0 / np.sqrt(self.num_features * self.num_times),
                                         homoNoise=self.homoNoise,
                                         random_num=init_repeats,
                                         phiNorm=np.sqrt(self.h),
                                         vNorm=1.0,
                                         random_state=random_state)
            initializer.run()
            print("initialization done.")
            self.Phi = initializer.Phi.copy();
            self.V = initializer.V.copy();
            self.mu = initializer.U.copy();
            self.sigma_mu = initializer.s2.copy();
            self.sigma_noise = initializer.sigma2.copy();
        if self.smooth_penalty is not None:
            if self.lam1_dfmin is None or self.lam1_dfmax:
                warnings.warn(
                    "Did not specify the range of dfs for modeling the trajectories."
                )
                self.lam1_dfmin = 1
                self.lam1_dfmax = self.num_times-2
        self.check_init()
    def posteriorU(self, meanOnly = False):
        PhiV = np.zeros((self.num_times * self.num_features, self.K))
        # O(KTJ)
        s2 = np.zeros((self.num_times * self.num_features))
        for j in np.arange(self.num_features):
            ll = np.arange(j * self.num_times,(j + 1) * self.num_times)
            s2[ll] = self.sigma_noise[j]
        for k in np.arange(self.K):
            PhiV[:, k] = np.kron(
                self.V[:, k].reshape(self.num_features, 1),
                self.Phi[:, k].reshape(
                    (self.num_times, 1))).reshape(-1)
        # O(I(|ni|K^2+JT)
        for i in np.arange(self.num_subjects):
            xte0 = self.intermediantes['Xmod1'][i, :]
            obs0 = self.intermediantes['O1'][i, :]
            obs0 = np.where(obs0 == 1)[0]
            PhiV0 = PhiV[obs0, :]
            xte0 = xte0[obs0]
            s20 = s2[obs0]
            #tmp1 = (V\cdot Phi)^t L^{-1} (V\cdot Phi)
            tmp1 = np.matmul(np.transpose(PhiV0) / s20,PhiV0)
            #tmp2 = (V\cdot Phi)^t L^{-1} X
            tmp2 = np.matmul(np.transpose(PhiV0),xte0 / s20)
            tmp3 = np.zeros((self.K))
            M = np.diag(1.0/self.sigma_mu)
            # tmp4 = (V\cdot Phi)^t L^{-1} (V\cdot Phi)+Lambda^{-1}
            tmp4 = tmp1 + M
            tmp4 = np.linalg.inv(tmp4)
            if not meanOnly:
                self.cov[i, :, :] = tmp4.copy()
            #M = 1/s^2 - cov_i/(s^4)=1/s^4(s^2-cov_i)
            M = M - np.matmul(np.matmul(M, self.cov[i,:,:]), M)
            if self.Z is not None:
                tmp3 = np.matmul(self.Z[i,],self.beta) / self.sigma_mu
                if self.fit_intercept:
                    tmp3 = tmp3 + self.intercepts / self.sigma_mu
            self.mu[i, :] = np.matmul(self.cov[i, :, :],tmp2 + tmp3)
            # transforma_mat documents M^i matrices in the supp
            self.transform_mat[i,:, :] = M.copy()
            # transforma_vec documents m^i vectors in the supp
            self.transform_vec[i,:] = np.matmul(np.matmul(tmp2, self.cov[i,:,:]),np.diag(1.0/self.sigma_mu))
    def V_update(self):
        #precalculation of large matrices used
        A0 = np.zeros((self.num_subjects * self.num_times, self.K))
        #A0: phi \odot u
        for k in np.arange(self.K):
            A0[:,k] = np.kron(self.Phi[:,k].reshape((self.num_times, 1)),
                              self.mu[:,k].reshape((self.num_subjects, 1))).reshape(-1)
        #cross0[k,j]= <XJ[j,:], A0[:,k]>/sigma[j]
        #cross1[k1,k2,j] = <A0[:,k1], A0[:,k2]>+sum_{t}<Sigma[,k1,k2]*Phi[t,k1]*Phi[t,k2]>
        cross0 = np.zeros((self.K, self.num_features))
        cross1 = np.zeros((self.K, self.K, self.num_features))
        for k in np.arange(self.K):
            for j in np.arange(self.num_features):
                x0 = self.intermediantes['Xmod2'][j,:]
                x1 = A0[:,k]
                obs0 = np.where(self.intermediantes['O2'][j, :] == 1)[0]
                cross0[k,j] = np.sum(x0[obs0] * x1[obs0])/self.sigma_noise[j]
                for l in np.arange(self.K):
                    if l <= k:
                        x0 = A0[:, k]
                        x1 = A0[:, l]
                        x3 = self.cov[:,k,l]
                        cross1[k,l,j] =  np.sum(x0[obs0] * x1[obs0])
                        for t in np.arange(self.num_times):
                            ii = np.arange(self.num_subjects)[self.O[:,t,j]==1]
                            cross1[k,l,j] += np.sum(x3[ii] * self.Phi[t,k] * self.Phi[t,l])
                        cross1[l,k,j] = cross1[k,l,j]
                cross1[:,:,j] = cross1[:,:,j]/self.sigma_noise[j]
        V0 = self.V.copy()
        for k in np.arange(self.K):
            a =cross1[k,k,:]
            b = np.zeros((self.num_features))
            b += cross0[k, :]
            for l in np.arange(self.K):
                if l != k:
                    b -= cross1[k, l, :] * self.V[:, l]
            laplace_lam = binary_search_vector(a, b, norm_constraint = 1.0)
            vnew = b/(a+laplace_lam)
            self.V[:,k] = vnew.copy()
        delta =  np.sum((V0- self.V)**2)
        return delta
    def smooth_penaty_update(self, a, b):
        A = np.diag(a)
        lam1sk = np.zeros((self.nlam1))
        dfs = self.lam1_dfmin + (1.0 - np.arange(self.nlam1) / (self.nlam1 - 1.0)) * (self.lam1_dfmax - 1.0)
        for l in np.arange(self.nlam1):
            df_hat, lam1sk[l] = penalty_search(A, self.Omega,
                df0=dfs[l],tol=1e-4, max_iter=100)
        #print(lam1sk)
        phi_best = b/(a+self.ridge_traj)
        errors = np.zeros((self.num_times, self.nlam1))
        for l in np.arange(self.nlam1):
            phi0 = np.linalg.solve(np.diag(a+self.ridge_traj) + lam1sk[l] * self.Omega, b)
            Hdiag = np.diag(np.linalg.inv(
                np.diag(a+self.ridge_traj) + lam1sk[l] * self.Omega))
            Hdiag = Hdiag * ((a+self.ridge_traj))
            errors[:,l] = (a+self.ridge_traj)*(phi0 - phi_best) ** 2 /(1.0 - Hdiag) ** 2
        errors_mean = np.mean(errors, axis=0)
        errors_sd = np.sqrt(np.var(errors, axis=0) / self.num_times)
        idx1 = np.argmin(errors_mean)
        c = errors_mean[idx1] + errors_sd[idx1] * self.lam1criterion
        idx = np.where(errors_mean <= c)[0]
        idx = np.max(idx)
        #print(lam1sk[idx])
        return lam1sk[idx]
    def Phi_update(self):
        A0 = np.zeros((self.num_subjects * self.num_features, self.K))
        # A0: v \odot u
        for k in np.arange(self.K):
            A0[:,k] =np.kron(self.V[:,k].reshape((self.num_features, 1)),\
                        self.mu[:,k].reshape((self.num_subjects, 1))).reshape(-1)
        s2 = np.zeros((self.num_subjects * self.num_features))
        for i in np.arange(self.num_subjects):
            ll = np.arange((i * self.num_features), (i+1)*self.num_features)
            s2[ll] = self.sigma_noise.copy()
        cross0 = np.zeros((self.K, self.num_times))
        cross1 = np.zeros((self.K, self.K, self.num_times))
        for k in np.arange(self.K):
            for t in np.arange(self.num_times):
                x0 = self.intermediantes['Xmod3'][t,:]
                x1 = A0[:,k]
                obs0 = np.where(self.intermediantes['O3'][t, :] == 1)[0]
                cross0[k,t] = np.sum(x0[obs0] * x1[obs0]/s2[obs0])
                for l in np.arange(self.K):
                    if l <= k:
                        x0 = A0[:, k]
                        x1 = A0[:, l]
                        x3 = self.cov[:,k,l]
                        cross1[k,l,t] =  np.sum(x0[obs0] * x1[obs0]/s2[obs0])
                        for j in np.arange(self.num_features):
                            ii = np.arange(self.num_subjects)[self.O[:,t,j]==1]
                            cross1[k,l,t] += np.sum(x3[ii] * self.V[j,k] * self.V[j,l]/s2[ii])
                        cross1[l,k,t] = cross1[k,l,t]
        # O(KITJ)
        Phi0 = self.Phi.copy()
        for k in np.arange(self.K):
            a = cross1[k, k, :]
            b = np.zeros((self.num_times))
            b += cross0[k, :]
            for l in np.arange(self.K):
                if l != k:
                    b -= cross1[k, l, :] * self.Phi[:, l]
            if self.smooth_penalty:
                self.lambda1[k] = self.smooth_penaty_update(a, b)
                phinew = phi_solver(a=a, b=b, Omega=self.Omega,
                           lam1=self.lambda1[k],
                           ridge_traj=self.ridge_traj, h=np.sqrt(self.h))
            else:
                phinew = phi_solver(a=a, b=b,
                                    Omega=self.Omega,
                                    lam1=0.0,
                                    ridge_traj=self.ridge_traj,
                                    h=np.sqrt(self.h))
            self.Phi[:, k] = phinew.copy()
        delta = np.sum((Phi0 - self.Phi) ** 2) / (self.h * self.h)
        return delta
    def sigma_noise_update(self):
        sigma_noise = np.zeros((self.num_features))
        A0 = np.zeros((self.num_subjects * self.num_times, self.K))
        A1 = np.zeros((self.K, self.K))
        for k in np.arange(self.K):
            A0[:,k] = np.kron(self.mu[:,k].reshape((self.num_subjects,1)),\
                        self.Phi[:,k].reshape((self.num_times,1))).reshape(-1)
        obs0 = np.where(self.intermediantes['O2'][0, :]==1)[0]
        for k in np.arange(self.K):
            for l in np.arange(self.K):
                if l <= self.K:
                    A1[k,l] = np.sum(A0[obs0,k] * A0[obs0,l])
                    x1 = self.cov[:,k,l]
                    for t in np.arange(self.num_times):
                        ii = np.arange(self.num_subjects)[self.O[:, t, 0] == 1]
                        A1[k, l] += np.sum(x1[ii] * self.Phi[t, k] * self.Phi[t, l])
                    A1[l, k] = A1[k,l]
        tmp1 = 0.0
        tmp2 = 0.0
        for j in np.arange(self.num_features):
            x = self.intermediantes['Xmod2'][j,:]
            x1 = np.matmul(A0, self.V[j,:])
            tmp = np.sum(np.matmul(self.V[j,].reshape((1,self.K)), A1) * self.V[j,:])
            tmp3 = (np.sum(x[obs0]**2) - 2 * np.sum(x[obs0] * x1[obs0]) + tmp)
            tmp4 = float(len(obs0))
            sigma_noise[j] = tmp3/tmp4
            tmp1 += tmp3
            tmp2 += tmp4
        if self.homoNoise:
            self.sigma_noise[:] = tmp3/tmp4
    def sigma_mu_update(self):
        sigma_mu = np.zeros((self.K))
        fitted = np.zeros((self.num_subjects, self.K))
        if self.Z is not None:
            fitted = np.matmul(self.Z, self.beta)
            if self.fit_intercept:
                for k in np.arange(self.K):
                    fitted[:,k] += self.intercepts[k]
        for i in np.arange(self.num_subjects):
           tmp = (self.mu[i, :] - fitted[i,:])
           sigma_mu += tmp **2
           sigma_mu += np.diag(self.cov[i,:,:])
        self.sigma_mu = sigma_mu / float(self.num_subjects)
    def beta_update(self):
        if self.lasso_penalty:
            self.beta, self.intercepts, self.lambda2 = beta_fit(Z = self.Z, transform_mat = self.transform_mat,
                     transform_vec = self.transform_vec, beta = self.beta,
                     intercepts = self.intercepts, lambda2 = self.lambda2, fit_intercept=self.fit_intercept,
                     lam2_update=self.lam2_update, nlam2=self.nlam2, nfolds=self.beta_folds,
                     max_iter=1, tol=0.01)
        else:
            self.beta, self.intercepts, self.lambda2 = beta_fit(Z = self.Z, transform_mat = self.transform_mat,
                     transform_vec = self.transform_vec, beta = self.beta,
                     intercepts = self.intercepts, lambda2 = self.lambda2, fit_intercept=self.fit_intercept,
                     lam2_update=self.lam2_update, nlam2=self.nlam2, nfolds=self.beta_folds,
                     max_iter=1, tol=0.01)



class SPACO(SPACObase):
    def __init__(self, data_obj):
        super().__init__(data_obj)
    def reordering(self):
        #order the CP vector and flip signs
        mags = self.mu.std(axis = 0)
        idx = (-mags).argsort()
        self.mu = self.mu[:,idx]
        self.cov = self.cov[:,idx,:][:,:,idx]
        self.V = self.V[:,idx]
        self.Phi = self.Phi[:,idx]
        self.sigma_mu=self.sigma_mu[idx]
        self.transform_mat = self.transform_mat[:,idx,:][:,:,idx]
        self.transform_vec = self.transform_vec[:, idx]
        self.lambda2 =self.lambda2[idx]
        self.lambda1 = self.lambda1[idx]
        if self.beta is not None:
            self.beta = self.beta[:,idx]
            self.intercepts = self.intercepts[idx]
    def train(self, update_cov = True,
              update_sigma_mu=True, update_sigma_noise=True,
              lam1_update=True, lam2_update=True,
              max_iter: int = 100, min_iter: int =1,
              tol = 1e-6, trace = True,
              reordering = True
              ):
        err = 2 * tol
        iter = 0;
        self.lam1_update = lam1_update
        self.lam2_update = lam2_update
        self.posteriorU()
        while (iter < max_iter and err > tol) or iter < min_iter:
            if trace:
                print("iteration " + str(iter) + " : " + str(err))
            delta2 = self.Phi_update()
            delta1 = self.V_update()
            self.posteriorU(meanOnly=(not update_cov))
            if self.Z is not None:
                # update beta
                self.beta_update()
            if update_sigma_mu:
                self.sigma_mu_update()
            if update_sigma_noise:
                self.sigma_noise_update()
            self.reordering()
            err = delta1 + delta2
            iter = iter + 1

class SPACOcv(SPACO):
    def __init__(self, data_obj):
        super().__init__(data_obj)
        self.train_ids = None
        self.test_ids = None
        self.nfolds = None
        self.crossV = None
        self.crossPhi = None
        self.cross_sigma_mu = None
        self.mu_cross = None
        self.cov_cross =None
        self.transform_mat_cross = None
        self.transform_vec_cross =None
        self.crossIntercept = None
        self.crossBeta = None
        self.crossIntercept = None
        self.s2 = np.zeros((self.num_features * self.num_times))
        self.cross_likloss = None
        self.testing_input = dict()
        self.testing_output = dict()
    def cross_validation_train(self,
                               train_ids,
                               test_ids,
                               max_iter=10,
                               min_iter = 1,
                               tol=1e-3,
                               trace = True):
        for j in np.arange(self.num_features):
            ll = np.arange(j * self.num_times,(j + 1) * self.num_times)
            self.s2[ll] = self.sigma_noise[j]
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.nfolds = len(self.train_ids)
        self.crossV = np.zeros((self.num_features, self.K, self.nfolds))
        self.crossPhi = np.zeros((self.num_times, self.K, self.nfolds))
        self.cross_sigma_mu = np.zeros((self.K, self.nfolds))
        self.mu_cross = self.mu.copy()
        self.cov_cross = self.cov.copy()
        self.transform_mat_cross = self.transform_mat.copy()
        self.transform_vec_cross = self.transform_vec.copy()
        if self.Z is not None:
            self.crossBeta = np.zeros((self.Z.shape[1], self.K, self.nfolds))
            self.crossIntercept = np.zeros((self.K, self.nfolds))
        else:
            self.crossBeta = None
            self.crossIntercept = None

        for fold_id in np.arange(self.nfolds):
            data_obj_cv = dict(X=self.X,
                           O = self.O,
                           Z = self.Z,rank=self.K,
                           time_stamps = self.time_stamps)
            # initialize parameters from full fit
            cv_obj = SPACO(data_obj_cv)
            data_entires = [a for a in dir(cv_obj) if not a.startswith( '__') and not callable(getattr(cv_obj, a))]
            for i in range(len(data_entires)):
                tmp = getattr(self, data_entires[i])
                setattr(cv_obj, data_entires[i],copy.deepcopy(tmp))
            #modify terms that need to be changed
            cv_obj.X0 = cv_obj.X0[self.train_ids[fold_id], :,:]
            cv_obj.X = cv_obj.X[self.train_ids[fold_id],:,:]
            cv_obj.O = cv_obj.O[self.train_ids[fold_id],:,:]
            cv_obj.mu = cv_obj.mu[self.train_ids[fold_id],:]
            cv_obj.cov = cv_obj.cov[self.train_ids[fold_id],:,:]
            if self.Z is not None:
                cv_obj.Z = cv_obj.Z[self.train_ids[fold_id],:]
            cv_obj.num_subjects = len(self.train_ids[fold_id])
            cv_obj.intermediantes['Xmod1'], \
            cv_obj.intermediantes['O1'], \
            cv_obj.intermediantes['Xmod2'], \
            cv_obj.intermediantes['O2'], \
            cv_obj.intermediantes['Xmod3'], \
            cv_obj.intermediantes['O3'] \
                = unfold(cv_obj.X, cv_obj.O)
            cv_obj.transform_mat = np.zeros(
                (cv_obj.num_subjects, cv_obj.K, cv_obj.K))
            cv_obj.transform_vec = np.zeros(
                (cv_obj.num_subjects, cv_obj.K))
            cv_obj.check_init()
            cv_obj.train(update_cov=True,
                       update_sigma_mu=True,
                       update_sigma_noise=False,
                       lam1_update=False, lam2_update=False,
                       max_iter=max_iter, min_iter=min_iter,
                       tol=tol, trace=trace, reordering = False)
            #get cross-fitted parameters
            self.crossV[:, :, fold_id] = cv_obj.V.copy()
            self.crossPhi[:, :, fold_id] = cv_obj.Phi.copy()
            self.cross_sigma_mu[:,fold_id] = cv_obj.sigma_mu.copy()
            if self.Z is not None:
                self.crossBeta[:, :, fold_id] = cv_obj.beta.copy()
                if self.fit_intercept is not None:
                    self.crossIntercept[:,fold_id] = cv_obj.intercepts.copy()
            #self.mu_cross = self.mu.copy()
            # correction
            xte = self.intermediantes['Xmod1'][self.test_ids[fold_id], :]
            ote = self.intermediantes['O1'][self.test_ids[fold_id], :]
            if self.Z is not None:
                zte = self.Z[self.test_ids[fold_id], :]
            else:
                zte = None
            PhiV =np.zeros((self.num_times*self.num_features, self.K))
            #find the most correlated PhiV dimension and flip the sign
            for k in np.arange(self.K):
                PhiV[:,k] = np.kron(cv_obj.V[:,k].reshape((self.num_features,1)),
                                     cv_obj.Phi[:,k].reshape((self.num_times,1))).reshape(-1)
            mu, cov,  transform_vec, transform_mat, temp_vec0 = \
                cross_posterior(xte = xte, ote = ote, zte = zte,
                                s2 = self.s2, sigma_mu= cv_obj.sigma_mu,
                                PhiV = PhiV, beta=cv_obj.beta,
                                intercepts=cv_obj.intercepts,
                                fit_intercept=cv_obj.fit_intercept)

            self.transform_vec_cross[self.test_ids[fold_id],:] = transform_vec.copy()
            self.transform_mat_cross[self.test_ids[fold_id],:,:] = transform_mat.copy()
            self.mu_cross[self.test_ids[fold_id],:] = mu.copy()
            self.cov_cross[self.test_ids[fold_id],:,:] = cov.copy()
            if self.Z is not None:
                self.crossBeta[:, :, fold_id] = cv_obj.beta.copy()
                self.crossIntercept[:,fold_id] = cv_obj.intercepts.copy()

    def cross_logliklihood(self):
        #caclulate log likelihood using cross validation for rank k=1,...,K
        loglik = np.zeros((self.num_subjects, self.K))
        s2 = np.zeros((self.num_times * self.num_features))
        for j in np.arange(self.num_features):
            ll = np.arange(j * self.num_times,(j + 1) * self.num_times)
            s2[ll] = self.sigma_noise[j]
        for k in np.arange(len(self.test_ids)):
            Phi = self.crossPhi[:,:,k]
            V =self.crossV[:,:,k]
            sigma_mu = self.cross_sigma_mu[:,k]
            cov = self.cov_cross
            beta = self.crossBeta[:,:,k]
            mu = np.matmul(self.Z, beta)
            PhiV =np.zeros((self.num_times*self.num_features, self.K))
            #find the most correlated PhiV dimension and flip the sign
            for l in np.arange(self.K):
                PhiV[:,l] = np.kron(V[:,l].reshape((self.num_features,1)),
                                     Phi[:,l].reshape((self.num_times,1))).reshape(-1)
            for i in self.test_ids[k]:
                xte0 = self.intermediantes['Xmod1'][i, :]
                obs0 = self.intermediantes['O1'][i, :]
                obs0 = np.where(obs0 == 1)[0]
                PhiV0 = PhiV[obs0, :]
                xte0 = xte0[obs0]
                s20 = s2[obs0]
                xhat = np.zeros(xte0.shape)
                det_sigmmu= 1.0
                for l in np.arange(self.K):
                    det_sigmmu *= sigma_mu[l]
                    xhat += PhiV0[:,l] * mu[i, l]
                    fi = xte0-xhat
                    loglik[i,l] = np.sum(fi **2/s20)
                    tmp1 = np.matmul(fi/s20, PhiV0[:,:(l+1)])
                    Sigmai =cov[i,:(l+1),:(l+1)]
                    loglik[i, l] -= np.sum(np.matmul(tmp1, Sigmai) * tmp1)
                    loglik[i, l] += np.log(det_sigmmu) - np.log(np.linalg.det(cov[i,:(l+1),:(l+1)]))
                    loglik[i, l] += np.sum(np.log(s20))
        self.cross_likloss = loglik


'''
CRtest and CRtest_cross fix estiamted model parameters other than beta
and compare the p-values using randomized (conditional) variables.
'''
class CRtest_cross:
    def __init__(self, spaco_cross, type = 'cross', delta = 0.0):
        self.type = type
        self.K = spaco_cross.K
        self.delta = delta
        self.Z = spaco_cross.Z.copy()
        self.U = np.zeros((spaco_cross.num_subjects, spaco_cross.K))
        self.Ztmp = self.Z.copy()
        if self.type == 'cross':
            self.cov = spaco_cross.cov_cross.copy()
            self.sigma_mu = spaco_cross.cross_sigma_mu.copy()
            self.transform_mat = spaco_cross.transform_mat_cross.copy()
            self.transform_vec = spaco_cross.transform_vec_cross.copy()
            self.transform_vec = spaco_cross.transform_vec_cross.copy()
            self.test_ids = spaco_cross.test_ids
            self.train_ids = spaco_cross.train_ids
        else:
            self.cov = spaco_cross.cov.copy()
            self.transform_mat = spaco_cross.transform_mat.copy()
            self.transform_vec = spaco_cross.transform_vec.copy()
            self.test_ids = [np.arange(self.Z.shape[0])]
            self.train_ids = [np.arange(self.Z.shape[0])]
            self.sigma_mu = np.ones((self.K, 1))
            self.sigma_mu[:,0] = spaco_cross.sigmaF.copy()

        self.beta = spaco_cross.beta.copy()
        self.beta_tmp = self.beta.copy()
        self.beta_drop = np.zeros((spaco_cross.beta.shape[0], spaco_cross.beta.shape[1], spaco_cross.beta.shape[0]))
        self.intercepts = spaco_cross.intercepts.copy()
        self.intercepts_tmp = spaco_cross.intercepts.copy()
        self.intercepts_drop = np.zeros((self.K, self.beta.shape[0]))
        self.lambda2 = spaco_cross.lambda2.copy()
        self.fit_intercept = spaco_cross.fit_intercept
        self.lam2_update = spaco_cross.lam2_update
        self.nlam2 =  spaco_cross.nlam2

        self.foldid = None
        self.Zuse = np.zeros((self.Z.shape[0], self.K))
        self.Ztheta = np.zeros((self.Z.shape[0], self.K, self.K))
        self.offsets = np.ones((self.Z.shape[0], self.K))
        self.coef_marginal = np.zeros((self.Z.shape[1], self.K))
        self.coef_partial = np.zeros((self.Z.shape[1], self.K))
        self.residuals = np.zeros((self.Z.shape[0], self.K))
        self.cov_delta = self.cov.copy()
        self.noise_delta = np.ones((self.Z.shape[0], self.K, self.K))
        self.noise_inv = np.ones((self.Z.shape[0], self.K, self.K))
        self.prevec = self.transform_vec.copy()
        self.transform_y = np.zeros((self.Z.shape[0], self.K))
        self.transform_residuals = np.zeros((self.Z.shape[0], self.K))
        self.theta = np.zeros((self.Z.shape[1],self.K))
        self.weights_w = np.zeros((self.Z.shape[0], self.K))
        self.weights_z = np.zeros((self.Z.shape[0], self.K))

        self.coef_partial_random = None
        self.coef_marginal_random = None

    def cut_folds(self, nfolds = 5, random_state = 1):
        k_fold = KFold(nfolds, shuffle=True, random_state=random_state)
        subject_ids = np.arange(self.Z.shape[0])
        split_id_obj = k_fold.split(subject_ids)
        train_ids = []
        test_ids = []
        for train_index, test_index in split_id_obj:
            train_ids.append(train_index.copy())
            test_ids.append(test_index.copy())
        self.foldid = np.zeros(self.Z.shape[0], dtype = int)
        for fold_id in np.arange(nfolds):
            self.foldid[test_ids[fold_id]] = fold_id + 1

    def precalculation(self):
        #calculate prevec
        for fold_id in np.arange(len(self.test_ids)):
            for i0 in np.arange(len(self.test_ids[fold_id])):
                i = self.test_ids[fold_id][i0]
                tmp1 = np.linalg.inv(self.cov[i, :, :])
                tmp2 = np.diag(1.0 / self.sigma_mu[:, fold_id])
                self.prevec[i, :] = np.matmul(tmp1,self.transform_vec[i,:] * self.sigma_mu[:,fold_id])
                self.cov_delta[i, :, :] = np.linalg.inv(tmp1 - tmp2 + self.delta* tmp2)
                self.noise_delta[i,:,:] = np.diag(self.sigma_mu[:,fold_id]) + (1.0 - 2*self.delta)*self.cov_delta[i,:,:]+\
                                             (self.delta**2-self.delta) * np.matmul(self.cov_delta[i,:,:]/self.sigma_mu[:,fold_id],self.cov_delta[i,:,:])
                self.noise_inv[i,:,:] = np.linalg.inv(self.noise_delta[i,:,:])
    def beta_fun_full(self, foldid = None, nfolds = 5, max_iter = 1, tol = 0.01, fixbeta0 = True):
        self.beta_tmp, intercepts_tmp, lambda2 = beta_fit_cleanup(Z = self.Z, beta = self.beta_tmp,
        intercepts = self.intercepts_tmp, delta =self.delta, prevec = self.prevec,
        cov = self.cov_delta, noiseCov = self.noise_delta, test_ids = self.test_ids,
        sigmaF = self.sigma_mu, lambda2 = self.lambda2,factor_idx=None, lam2_update=self.lam2_update, nlam2= self.nlam2,
        nfolds = nfolds, foldid = self.foldid, fit_intercept = self.fit_intercept, max_iter =max_iter, tol  =tol,
        beta_fix = self.beta, intercepts_fix = self.intercepts, iffix =fixbeta0)
    def beta_fun_one(self, nfolds, j = 0, max_iter = 1, tol = 0.01, fixbeta0 = True):
        idx = np.arange(self.Z.shape[1])
        idx = np.delete(idx, j)
        Z1 = np.delete(self.Z, j, axis = 1)
        if fixbeta0:
            beta1 = np.delete(self.beta, j, axis=0).copy()
            intercepts = self.intercepts.copy()
        else:
            beta1 = np.delete(self.beta_tmp, j,axis=0).copy()
            intercepts = self.intercepts_tmp.copy()
        for k in np.arange(self.K):
            if self.beta_tmp[j, k] == 0:
                self.beta_drop[:, k, j] = self.beta_tmp[:,k].copy()
                self.intercepts_drop[k, j] = self.intercepts_tmp[k].copy()
            else:
                beta1, intercepts, lambda2 =beta_fit_cleanup(Z = Z1, beta = beta1,
        intercepts = intercepts, delta = self.delta, prevec = self.prevec,
        cov = self.cov_delta, noiseCov = self.noise_delta, test_ids = self.test_ids,
        sigmaF = self.sigma_mu, lambda2 = self.lambda2,  factor_idx = [k],lam2_update=self.lam2_update, nlam2= self.nlam2,
        nfolds = nfolds, foldid =self.foldid, fit_intercept = self.fit_intercept, max_iter = max_iter,tol=tol,
        beta_fix = beta1, intercepts_fix = intercepts, iffix =True)
            self.beta_drop[idx, k, j] = beta1[:,k].copy()
            self.intercepts_drop[k,j] = intercepts[k].copy()
    def precalculation_response(self, j = 0):
        ###precalculate ``response" used for calculating marginal or partial correlation
        fitted =np.matmul(self.Z,self.beta_drop[:,:,j])
        for k in np.arange(self.K):
            fitted[:,k] = fitted[:,k]+self.intercepts_drop[k,j]
        for fold_id in np.arange(len(self.test_ids)):
            for i0 in np.arange(len(self.test_ids[fold_id])):
                i = self.test_ids[fold_id][i0]
                self.transform_y[i,:] = np.matmul(self.cov_delta[i,:,:], self.prevec[i,:])
                for k in np.arange(self.K):
                    for l in np.arange(self.K):
                        if l != k:
                            self.transform_y[i, k] = self.transform_y[i,k] + self.delta * self.cov[i,k,l] * fitted[i,l] /self.sigma_mu[l,fold_id]
                    self.residuals[i, k] = self.transform_y[i, k]  - (1.0 - self.cov[i,k,k]/self.sigma_mu[k,fold_id]) * fitted[i,k]
                    self.weights_w[i,k] = 1.0/self.noise_delta[i,k,k]
                    self.weights_z[i,k] = 1.0 - self.delta * self.cov_delta[i,k,k]/self.sigma_mu[k,fold_id]
    #def Ztheta_prepare(self):
    def Ztheta_calculate(self, j = 0):
        for fold_id in np.arange(len(self.test_ids)):
            for i0 in np.arange(len(self.test_ids[fold_id])):
                i = self.test_ids[fold_id][i0]
                self.Ztheta[i,:,:] = self.Ztmp[i,j] * (np.identity(self.K) - self.delta * self.cov_delta[i,:,:]/self.sigma_mu[:,fold_id])
    def coef_partial_fun(self,j, inplace = True):
        joint = False
        tstats = np.zeros(self.K)
        if joint:
            #scale the response and the covariate
            A = np.zeros((self.K, self.K))
            a = np.zeros((self.K))
            for i in np.arange(self.Z.shape[0]):
                tmp = np.matmul(np.transpose(self.Ztheta[i,:,:]),self.noise_inv[i,:,:])
                A += np.matmul(tmp,self.Ztheta[i,:,:])
                a += np.matmul(tmp, self.residuals[i,:])
            tstats = np.matmul(np.linalg.inv(A),a)
        else:
            for k in np.arange(self.K):
                tstats[k] = np.sum(self.residuals[:,k] * self.Ztmp [:,j] * self.weights_w[:,k]*self.weights_z[:,k])/np.sum(self.Ztmp[:,j] * self.Ztmp[:,j] * self.weights_w[:,k]*self.weights_z[:,k]**2)
        if inplace:
            self.coef_partial[j, :] = tstats.copy()
        else:
            return tstats
    def coef_marginal_fun(self,j, inplace = True):
        joint = False
        tstats = np.zeros(self.K)
        if joint:
            #scale the response and the covariate
            A = np.zeros((self.K, self.K))
            a = np.zeros((self.K))
            for i in np.arange(self.Z.shape[0]):
                tmp = np.matmul(np.transpose(self.Ztheta[i,:,:]),self.noise_inv[i,:,:])
                A += np.matmul(tmp,self.Ztheta[i,:,:])
                a += np.matmul(tmp, self.transform_y[i,:])
            tstats = np.matmul(np.linalg.inv(A),a)
        else:
            for k in np.arange(self.K):
                tstats[k] = np.sum(self.transform_y[:,k] * self.Ztmp[:,j] * self.weights_w[:,k]*self.weights_z[:,k])/np.sum(self.Ztmp[:,j] * self.Ztmp[:,j] * self.weights_w[:,k]*self.weights_z[:,k]**2)
        if inplace:
            self.coef_marginal[j, :] = tstats.copy()
        else:
            return tstats
    def coef_random_fun(self, Zrandom, j, type = "partial"):
        B = Zrandom.shape[2]
        for b in np.arange(B):
            self.Ztmp[:, j] = Zrandom[:, j, b]
            if type == "partial":
                if self.coef_partial_random is None:
                    print("please initiate coef_partial_random to desired dimension")
                    break;
                self.coef_partial_random[j,:,b] = \
                    self.coef_partial_fun(j=j,inplace=False).copy()
            else:
                if self.coef_marginal_random is None:
                    print("please initiate coef_partial_random to desired dimension")
                    break;
                self.coef_marginal_random[j,:,b] = \
                    self.coef_marginal_fun(j=j,inplace=False).copy()
        self.Ztmp = self.Z.copy()
    def pvalue_calculation(self, type = "partial",pval_fit = True, dist_name ='nct'):
        pvals_fitted = np.zeros((self.Z.shape[1], self.K))
        pvals_empirical = np.zeros((self.Z.shape[1], self.K))
        if not pval_fit:
            pvals_fitted[:,:] = np.nan
        for j in np.arange(pvals_fitted.shape[0]):
            print(j)
            for k in np.arange(pvals_fitted.shape[1]):
                if type == "partial":
                    nulls = self.coef_partial_random[j, k, :]
                    tstat = self.coef_partial[j, k]
                else:
                    nulls = self.coef_marginal_random[j, k, :]
                    tstat = self.coef_marginal[j, k]
                # remove infinite values
                idx = np.where(~np.isnan(nulls))[0]
                pvals_empirical[j, k] = (np.sum(np.abs(nulls[idx]) >= np.abs(tstat)) + 1.0) / (len(idx) + 1.0)
                if pval_fit:
                    pvals_fitted[j,k] = pvalue_fit(z=tstat,nulls=nulls[idx], dist_name=dist_name)
        return pvals_empirical, pvals_fitted









'''
@spaco: a spaco/spaco_cross object
@Zconditional: randomly generated N * q * B array for conditional randomization test. None = skip.
@Zmarginal: randomly generated N * q * B array for conditional randomization test. None = skip.
@dist_name: distribution used for fitting the nulls.
@method: if spaco is from cross-fitting.
@trace: if print out the progress.
'''
