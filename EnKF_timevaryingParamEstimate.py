import math
import pandas as pd
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
import numpy as np
from numpy.linalg import inv
from numpy import dot
import scipy.stats as st
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def my_EnKF(Z, N, dim_state, dim_obs, forcing, param_ens, obs_noise):
    '''
    :param Z: the array of observation variables
    :param N: the number of samples in the ensemble
    :param dim_state: the number of the state variables
    :param dim_obs: the number of the observation variables
    :param forcing: the forcing data for NEE model
    :param param_ens: the ensemble of the parameter vectors
    :param obs_noise: the measurement noise
    :return: the estimated parameter vector, the updated parameter vector, the 95%-confidence interval of the updated parameter vector
    '''

    T = len(forcing)
    model = np.zeros((T, dim_state))
    anlys = np.zeros((T, dim_state))
    low_ci_bounds = np.zeros((T, dim_state))
    high_ci_bounds = np.zeros((T, dim_state))
    num_param = len(param_ens)

    for i in range(T):
        # prediction
        V_hat = forecast(N, param_ens, dim_state)
        model[i, :] = V_hat.mean(axis=1)
        # update
        R = np.eye(dim_obs) * obs_noise[i]
        Z_ens = generate_obs_ensemble(Z[i], R, N, dim_obs)
        update = analysis(V_hat, forcing[i, :], Z_ens, N, dim_state, dim_obs, R)

        # record target variables
        anlys[i, :] = update.mean(axis=1)
        param_ens = update[:num_param, :]

        low_ci_bound, high_ci_bound = st.t.interval(0.95, N - 1, loc=update.mean(axis=1), scale=st.sem(update, axis=1))
        low_ci_bounds[i, :] = low_ci_bound
        high_ci_bounds[i, :] = high_ci_bound

    return model, anlys, low_ci_bounds, high_ci_bounds


def nee_model(param_vect, driver_vect):
    T0 = -46.02
    Tref = 10
    k = 0.021
    E0 = 178

    Ta, Ts, Rad, VPD = driver_vect
    alpha, beta0, Rb = param_vect
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


def forecast(N, param_ens, dim_state):
    '''
    This function takes the posterior ensemble from the last step V_0, the state dimension dim_state,
    the number of ensemble members N, the current timestep t
    It uses the integrate function the create the forecast ensemble V_hat = forecast.
    '''

    forecast = np.zeros((dim_state, N))
    # 参数的随机游走模型不确定性用Q = diag((1/0.9975-1)*P_a)表示,Q是对角矩阵
    P_a = np.cov(param_ens)
    Q_param = np.diag(np.diag((1/0.9975-1)*P_a))
    for i in range(N):
        param_new = param_ens[:, i] + np.random.multivariate_normal(np.zeros(dim_state), Q_param, 1).flatten()
        forecast[:, i] = param_new
    return forecast


def generate_obs_ensemble(Z, R, N, dim_obs):
    '''
    This function takes the observation Z, the measurement noise expected value and variance measure_noise,
    the number of ensemble memebers N, and the number of observed components dim_obs.
    It returns the pertubations of the observation Z_new = z_ens.
    '''

    Z_new = np.zeros((dim_obs, N))
    for n in range(N):
        Z_new[:, n] = Z + np.random.multivariate_normal(np.zeros(dim_obs), R, 1)
    return Z_new


def analysis(V_hat, forcing, Z_ens, N, dim_state, dim_obs, R):
    '''
    This function takes the forecast ensemble V_hat, the forecast covariance matrix C, the observation matrix H,
    the ensemble of the observation Z_ens, the number of ensemble members K, the system dimension d, and the observation
    error covariance R.
    It returns the posterior ensemble V, saved into process.
    '''

    V = np.zeros((dim_state, N))

    Z_hat = np.zeros((dim_obs, N))
    for i in range(N):
        Z_hat[:, i] = nee_model(V_hat[:, i], forcing[:])

    anomaly_x = anomaly(V_hat, N)
    anomaly_y = anomaly(Z_hat, N)
    cov_xy = 1/(N-1) * dot(anomaly_x, np.transpose(anomaly_y))
    cov_yy = 1/(N-1) * dot(anomaly_y, np.transpose(anomaly_y))

    K = dot(cov_xy, inv(cov_yy + R))
    for n in range(N):
        V[:, n] = V_hat[:, n] - dot(K, Z_hat[:, n] - Z_ens[:, n])
    return V


def anomaly(samples, N):
    anomaly = np.zeros_like(samples)
    # 计算离均差
    for n in range(N):
        anomaly[:, n] = samples[:, n] - samples.mean(axis=1)

    return anomaly


def validation(param, driver):
    t = len(driver)
    result = np.zeros((t, 1))
    for i in range(t):
        if (driver[i, :] != -9999).all():
            result[i, :] = nee_model(param[i, :], driver[i, :])
        else:
            result[i, :] = np.nan
    return result


dim_obs = 1   # number of the observation variables
dim_state = 3   # the state dimension d
N = 500
# parameter vector: alpha, beta0, Rref, E0
# create initial ensemble from uniform distribution
# min = [0, 0, 0, 100]
# max = [0.1, 100, 20, 250]
# np.random.seed(0)
# param_ens = np.random.uniform(low=min, high=max, size=(N, dim_state))

# create initial ensemble from normal distribution
# different way to create initial ensemble only influence the start of the resulting time series and do not have much
# effect on the later part of the estimated time series
mean = np.array([0.04, 22.75, 2.91,])# 181.68])
cov = np.diag([0.00096, 295.35, 3.04,])# 4489.69])
np.random.seed(0)
param_ens = np.random.multivariate_normal(mean, cov, N)

param_ens = param_ens.T
# print(np.var(param_ens, axis=1))

data = pd.read_csv(r'G:\fluxnet2015\fluxnetUnpack\FLX_US-Los_FLUXNET2015_FULLSET_DD_2000-2014_2-4.csv',
                   index_col='TIMESTAMP')
var = data.loc['20030101':'20031231', ['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'VPD_F_MDS',
                                       'TA_F_MDS_QC', 'TS_F_MDS_1_QC', 'SW_IN_F_QC', 'VPD_F_MDS_QC',
                                       'NEE_VUT_USTAR50', 'NEE_VUT_USTAR50_QC', 'NEE_VUT_USTAR50_RANDUNC']]

var[var == -9999] = np.nan
var.dropna(axis=0, subset=['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'NEE_VUT_USTAR50'], inplace=True)

forcing = var.loc[:, ['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'VPD_F_MDS']].values
obs = var.loc[:, ['NEE_VUT_USTAR50']]
Z = obs.values
date_start = datetime.datetime(2003, 1, 1, 0, 0, 0)
date_end = datetime.datetime(2004, 1, 1, 0, 0, 0)

# obs_noise is variance, not standard deviation
obs_noise = var['NEE_VUT_USTAR50_RANDUNC'].fillna(var['NEE_VUT_USTAR50_RANDUNC'].mean()).values**2

starttime = datetime.datetime.now()
model, anlys, ci_low, ci_high = my_EnKF(Z, N, dim_state, dim_obs, forcing, param_ens, obs_noise)
endtime = datetime.datetime.now()
print('running time %s s' % (endtime-starttime).seconds)

# validate the performance of the estimated parameters
estimatedNEE = validation(anlys, forcing)
plt.plot(estimatedNEE, label = 'estimated NEE')
plt.plot(obs.values, label = 'observed NEE')
plt.legend()
plt.title('US-Los 2003')
plt.ylabel('NEE')
plt.xlabel('day')
plt.show()