import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from EnKF import EKI, validation


dim_obs = 1  # number of the observation variables
dim_param = 4  # the parameter dimension
N = 500
# parameter vector: alpha, beta0, Rref, E0
# create initial ensemble from uniform distribution
min = [0, 0, 0, 100]
max = [0.1, 50, 20, 200]
np.random.seed(0)
param_ens = np.random.uniform(low=min, high=max, size=(N, dim_param))

# create initial ensemble from normal distribution
# different way to create initial ensemble only influence the start of the resulting time series and do not have much
# effect on the later part of the estimated time series
# mean = np.array([0.04, 22.75, 2.91,  181.68])
# cov = np.diag([0.00096, 295.35, 3.04, 4489.69])
# np.random.seed(0)
# param_ens = np.random.multivariate_normal(mean, cov, N)

param_ens = param_ens.T
# print(np.var(param_ens, axis=1))

data = pd.read_csv(r'G:\fluxnet2015\fluxnetUnpack\FLX_US-Los_FLUXNET2015_FULLSET_HH_2000-2014_2-4.csv')
data.index = pd.to_datetime(data['TIMESTAMP_START'].astype('str'))

date_start = datetime.datetime(2003, 1, 1, 0, 0, 0)
date_end = datetime.datetime(2003, 12, 1, 0, 0, 0)

var = data.loc[date_start:date_end, ['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'VPD_F_MDS',
                                       'TA_F_MDS_QC', 'TS_F_MDS_1_QC', 'SW_IN_F_QC', 'VPD_F_MDS_QC',
                                       'NEE_VUT_USTAR50', 'NEE_VUT_USTAR50_QC', 'NEE_VUT_USTAR50_RANDUNC']]

var[var == -9999] = np.nan
# var.dropna(axis=0, subset=['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'NEE_VUT_USTAR50'], inplace=True)
# only estimate the parameters for observed NEE not for interpolated NEE
var.loc[var['NEE_VUT_USTAR50_QC'] > 0, 'NEE_VUT_USTAR50'] = np.nan

forcing = var.loc[:, ['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'VPD_F_MDS']].values
obs = var.loc[:, ['NEE_VUT_USTAR50']]
Z = obs.values

# obs_noise is variance, not standard deviation
# obs_noise = var['NEE_VUT_USTAR50_RANDUNC'].fillna(var['NEE_VUT_USTAR50_RANDUNC'].mean()).values ** 2
obs_noise = var['NEE_VUT_USTAR50_RANDUNC'].mean() ** 2
H = np.array([[0, 0, 0, 0, 1]])
starttime = datetime.datetime.now()
model, anlys, anlys_std = EKI(Z, H, N, dim_param, dim_obs, forcing, param_ens, obs_noise)
endtime = datetime.datetime.now()
print('running time %s s' % (endtime - starttime).seconds)

prior = pd.DataFrame(model, index=obs.index, columns=['alpha', 'beta0', 'Rref', 'E0', 'NEE'])
posterior = pd.DataFrame(anlys, index=obs.index, columns=['alpha', 'beta0', 'Rref', 'E0', 'NEE'])

# validate the performance of the estimated parameters


estimatedNEE = validation(anlys[:-1], forcing)
plt.plot(estimatedNEE, label='estimated NEE')
plt.plot(obs.values, label='observed NEE')
plt.legend()
plt.title('US-Los 2003')
plt.ylabel('NEE')
plt.xlabel('day')
plt.show()
