import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from EnKF import EKI, validation, forecast, update
import math
import random

def nee_model(param_vect, driver_vect):
    T0 = -46.02
    Tref = 15
    k = 0.021
    # E0 = 178
    Ta, Ts, Rad, VPD = driver_vect
    alpha, beta0, Rb, E0 = param_vect
    if Ts > 0:
        r = Rb * math.exp(E0 * (1 / (Tref - T0) - 1 / (Ta - T0)))
    else:
        r = Rb * math.exp(E0 * (1 / (Tref - T0) - 1 / (Ts - T0)))
    if Rad > 4:
        if VPD > 10:
            beta = beta0 * np.exp(-k*(VPD-10))
        else:
            beta = beta0
        nee = -1 * alpha * beta * Rad / (alpha * Rad + beta) + r
    else:
        nee = r
    return nee


def NEE_model_con(param, forcing):
    t0 = 227.13
    e0 = 185.45
    a, k, rp = param
    ta, ppfd, ts = forcing
    if ts > 0:
        r = rp * np.exp(-1 * e0 / (ta + 273.15 - t0))
        if ppfd < 5:
            nee = r
        else:
            nee = (-1 * a * ppfd) / (k + ppfd) + r
    else:
        r = rp * np.exp(-1 * e0 / (ts + 273.15 - t0))
        if ppfd < 5:
            nee = r
        else:
            nee = a
    return nee


# dim_obs = 1  # number of the observation variables
# dim_param = 4  # the parameter dimension
# N = 50
# # parameter vector: alpha, beta0, Rref, E0
# # create initial ensemble from uniform distribution
# # min = [0, 0, 0, 150]
# # max = [0.1, 50, 20, 200]
# # np.random.seed(0)
# # param_ens = np.random.uniform(low=min, high=max, size=(N, dim_param))
#
# # create initial ensemble from normal distribution
# # different way to create initial ensemble only influence the start of the resulting time series and do not have much
# # effect on the later part of the estimated time series
# mean = np.array([0.04, 22.75, 2.91, 181.68])
# cov = np.diag([0.00096, 331, 3.04, 21124.87])
# # cov = np.array([[0.00096, 0, 0, 0],
# #                 [0, 295.35, 0, 45.025],
# #                 [0, 0, 3.04, 0],
# #                 [0, 0, 45.025, 21124.87]
# #                 ])
# np.random.seed(0)
# param_ens = np.random.multivariate_normal(mean, cov, N)
#
# param_ens = param_ens.T
# print(param_ens.mean(axis=1).round(3))
# print(np.var(param_ens, axis=1).round(3))
#
# data = pd.read_csv(r'G:\fluxnet2015\fluxnetUnpack\FLX_US-Los_FLUXNET2015_FULLSET_HH_2000-2014_2-4.csv')
# data.index = pd.to_datetime(data['TIMESTAMP_START'].astype('str'))
#
# date_start = datetime.datetime(2003, 6, 1, 0, 0, 0)
# date_end = datetime.datetime(2003, 12, 1, 0, 0, 0)
#
# var = data.loc[date_start:date_end, ['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'VPD_F_MDS',
#                                      'TA_F_MDS_QC', 'TS_F_MDS_1_QC', 'SW_IN_F_QC', 'VPD_F_MDS_QC',
#                                      'NEE_VUT_USTAR50', 'NEE_VUT_USTAR50_QC', 'NEE_VUT_USTAR50_RANDUNC']]
#
# var[var == -9999] = np.nan
# # var.dropna(axis=0, subset=['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'NEE_VUT_USTAR50'], inplace=True)
# # only estimate the parameters for observed NEE not for interpolated NEE
# var.loc[var['NEE_VUT_USTAR50_QC'] > 0, 'NEE_VUT_USTAR50'] = np.nan
#
# forcing = var.loc[:, ['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'VPD_F_MDS']].values
# obs = var.loc[:, ['NEE_VUT_USTAR50']]
# Z = obs.values
#
# # obs_noise is variance, not standard deviation
# # obs_noise = var['NEE_VUT_USTAR50_RANDUNC'].fillna(var['NEE_VUT_USTAR50_RANDUNC'].mean()).values ** 2
# obs_noise = var['NEE_VUT_USTAR50_RANDUNC'].mean() ** 2
# H = np.array([[0, 0, 0, 0, 1]])
# starttime = datetime.datetime.now()
# model, anlys, anlys_std = EKI(Z, nee_model, H, N, dim_param, dim_obs, forcing, param_ens, obs_noise)
# endtime = datetime.datetime.now()
# print('running time %s s' % (endtime - starttime).seconds)
#
# prior = pd.DataFrame(model, index=obs.index, columns=['alpha', 'beta0', 'Rref', 'E0', 'NEE'])
# posterior = pd.DataFrame(anlys, index=obs.index, columns=['alpha', 'beta0', 'Rref', 'E0', 'NEE'])
#
# # validate the performance of the estimated parameters
#
#
# estimatedNEE = validation(nee_model, anlys[:-1], forcing)
# plt.plot(obs.values, label='observed NEE', colour='orange')
# plt.plot(estimatedNEE, label='estimated NEE', colour='b')
# plt.legend()
# plt.title('US-Los 2003')
# plt.ylabel('NEE')
# plt.xlabel('day')
# plt.show()


#######################################A test for EKI using sin function###############################################
seed = random.random()
def sin(A, v, forcing):
    phi = 2 * np.pi * seed
    return A*np.sin(forcing + phi) + v


def G(param, forcing):
    A, v = param
    sincurve = sin(A, v, forcing)
    return [np.max(sincurve)-np.min(sincurve), np.mean(sincurve)]


# create pseudo-observation
dim_obs = 2
gamma = 0.1*np.eye(dim_obs)
theta_true = [1.0, 7.0]
print(f'True parameter vector is {theta_true}\n')
t = np.arange(0, 2*np.pi+0.01, 0.01)
y = G(theta_true, t) + np.random.multivariate_normal(np.zeros(dim_obs), gamma)
y = y.reshape(2, 1)
print(f'noisy observation is {y}\n')
# solve the inverse problem
N_ens = 5
N_iter = 10
dim_param = 2
mean = [2, 0]
cov = np.diag([1, 5])
param_ens = np.random.multivariate_normal(mean, cov, N_ens).T
H = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1]])

for j in range(N_iter):
    V_hat = forecast(G, t, N_ens, param_ens, dim_param, dim_obs, var_inflation=False)
    C_hat = np.cov(V_hat, rowvar=True)
    analysis = update(V_hat, C_hat, H, y, N_ens, dim_param, dim_obs, gamma)
    param_ens = analysis[:dim_param, :]
    print(f'{j} iter resulting augmented state vector is {analysis.mean(axis=1)}')


print(f'finished {N_iter} iterations')
# print(f'At {i} time step, the EKI updated parameters are : {record_iter_results.mean(axis=(0, 2))}')
