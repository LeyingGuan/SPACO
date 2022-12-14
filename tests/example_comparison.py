import spaco
import numpy as np
import sys
import random
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

it = 211
I = 100; T = 30; J = 100; q = 5;
SNR2 = 5.0; SNR1 = 2.0
spaco.seed_everything(seed=it)

data = dataGen(I=I, T=T, J=J, q=q, rate = .8, s=3, K0 = 3, SNR1 = SNR1, SNR2 = SNR2)
rank = 3

comparison_res = spaco.comparison_pipe(Phi0=data[5],
                                 V0 = data[6],
                                 U0 = data[7],
                                 X = data[2],
                                 Z =  data[9],
                                 O = data[3],
                                 time_stamps = data[4])

comparison_res.compare_run(rank = 3, max_iter = 30)

for method in comparison_res.eval_dict.keys():
    if method != "empirical":
        print(method)
        print(comparison_res.eval_dict[method].alignment)

comparison_res.eval_dict['empirical']


