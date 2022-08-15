import spaco as spaco
import importlib
importlib.reload(spaco)
import numpy as np
import pandas as pd
import copy
def dataGen(I, T, J, q, rate, s=3, K0 = 3, SNR1 = 1.0, SNR2 = 3.0):
    Phi0 = np.zeros((T, K0))
    Phi0[:,0] = 1.0
    Phi0[:,1] = np.arange(T)/T
    Phi0[:, 1] = np.sqrt(1-Phi0[:,1]**2)
    Phi0[:,2] = (np.cos((np.arange(T))/T * 4*np.pi))
    for k in np.arange(K0):
        Phi0[:, k] = Phi0[:, k]
        Phi0[:, k] = Phi0[:, k] /(np.sqrt(np.mean(Phi0[:, k] ** 2))) * (np.log(J)+np.log(T))/np.sqrt(I*T*rate) *SNR1
    V0 = np.random.normal(size=(J, K0))*1.0/np.sqrt(J)
    Z = np.random.normal(size =(I, q))
    U = np.random.normal(size =(I,K0))
    beta = np.zeros((q,K0))
    for k in np.arange(K0):
        if q > 0:
            if s > q:
                s = q
            beta[:s,k] = np.random.normal(size = (s)) * np.sqrt(np.log(q)/I) * SNR2
            U[:,k] = U[:,k]+np.matmul(Z, beta[:,k])
        U[:,k] = U[:,k] - np.mean(U[:,k])
        U[:,k] = U[:,k]/np.std(U[:,k])
    Xcomplete = np.random.normal(size=(I, T, J)) * 1.0
    T0 = np.arange(T)
    signal_complete = np.zeros(Xcomplete.shape)
    PhiV0 = np.zeros((T, J, K0))
    for k in np.arange(K0):
        PhiV0[:,:,k] = np.matmul(Phi0[:,k].reshape((T,1)), V0[:,k].reshape(1,J))
    for i in np.arange(I):
        for k in np.arange(K0):
            signal_complete[i, :, :] += PhiV0[:,:,k] * U[i,k]
        Xcomplete[i, :, :] += signal_complete[i, :, :]
    Obs = np.ones(Xcomplete.shape, dtype=int)
    Xobs = Xcomplete.copy()
    for i in np.arange(I):
        ll = T0
        tmp = np.random.choice(T0,replace=False,size=T - int(rate * T))
        Obs[i, ll[tmp], :] = 0
        Xobs[i, ll[tmp], :] = np.nan
    return Xcomplete, signal_complete , Xobs, Obs, T0, Phi0, V0, U, PhiV0, Z, beta

it = 101
I = 100; T = 30; J = 10; q = 100;
SNR2 = 10.0; SNR1 = 1.0; rate = 0.1
spaco.seed_everything(seed=it)

data = dataGen(I=I, T=T, J=J, q=q, rate = rate, s=3, K0 = 3, SNR1 = SNR1, SNR2 = SNR2)

ranks = np.arange(1,11)
negliks = spaco.rank_selection_function(X = data[2], O = data[3], Z = data[9],
                                        time_stamps = data[4], ranks=ranks, early_stop = True,
                            max_iter = 30, cv_iter = 5, add_std = 0.0)


means = negliks.mean(axis = 0)
means_std  = means+negliks.std(axis = 0)/np.sqrt(I)*0.5
means=means[~np.isnan(means)]
means_std =means_std[~np.isnan(means_std)]
idx_min = np.argmin(means)
rank_min = ranks[idx_min]
rank_std= ranks[np.where(means<=means_std[idx_min])][0]
print(rank_min)
print(rank_std)

