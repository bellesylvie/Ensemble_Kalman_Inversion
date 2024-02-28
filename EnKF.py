import numpy as np
from numpy.linalg import inv
from numpy import dot
import scipy.stats as st

def EKI(Z, MODEL, H, N, dim_param, dim_obs, forcing, param_ens, obs_noise):
    '''
    :param Z: the array of observations
    :param N: the number of samples in the ensemble
    :param dim_param: the number of parameters to estimate
    :param dim_obs: the number of the observation variables
    :param forcing: the forcing data for NEE model, like temperature, solar radiation, etc.
    :param param_ens: the ensemble of the parameter vectors
    :param obs_noise: the measurement noise
    :return: the estimated parameter vector, the updated parameter vector, and the standard deviation of the updated
    parameters
    '''

    R = np.diag(abs(Z)*0.1)
    for j in range(10):
        V_hat = forecast(MODEL, forcing, N, param_ens, dim_param, dim_obs, var_inflation=False)
        C_hat = np.cov(V_hat, rowvar=True)
        analysis = update(V_hat, C_hat, H, Z, N, dim_param, dim_obs, R)
        param_ens = analysis[:dim_param, :]
    return [analysis.mean(axis=1), analysis.std(axis=1)]


def forecast(MODEL, forcing, N, param_ens, dim_param, dim_obs, var_inflation):
    '''
    This function takes the posterior ensemble from the last step V_0, the parameter dimension dim_param,
    the number of ensemble members N, the forcing data for the NEE model forcing, the dimension of observation dim_obs,
    and whether to do variance inflation var_inflation
    return: augmented state vector ensemble with parameters and observations
    '''

    forecast = np.zeros((dim_param+dim_obs, N))

    if ~var_inflation:
        param_ens_forecast = param_ens
    else:
        # 0.9975
        param_cov_infl = (1/0.90-1)*np.diag(np.diag(np.cov(param_ens, rowvar=True)))
        param_ens_forecast = np.zeros_like(param_ens)
        for i in range(N):
             param_ens_forecast[:, i] = param_ens[:, i] + np.random.multivariate_normal(np.zeros(dim_param), param_cov_infl, 1)

    forecast[:dim_param, :] = param_ens_forecast
    for i in range(N):
        forecast[dim_param:, i] = MODEL(param_ens_forecast[:, i], forcing)
    return forecast


def update(V_hat, C, H, Z, N, dim_param, dim_obs, R):
    '''
    This function takes the forecast ensemble V_hat, the forecast covariance matrix C, the observation matrix H,
    the ensemble of the observation Z_ens, the number of ensemble members N, the parameter dimension dim_param, and the
    number of observed components dim_obs.
    It returns the posterior ensemble V.
    '''
    V = np.zeros((dim_param+dim_obs, N))
    K = dot(dot(C, np.transpose(H)), inv(R + dot(H, dot(C, np.transpose(H)))))
    # print(f'Kalman gain is {K.round(3)}')
    for k in range(N):
        Z_perturb = Z + np.random.multivariate_normal(np.zeros(dim_obs), R, 1).T
        V[:, k] = (V_hat[:, k].reshape(dim_obs+dim_param, 1) - dot(K, dot(H, V_hat[:, k].reshape(dim_obs+dim_param, 1)) - Z_perturb)).flatten()
    return V


def validate(MODEL, param, driver):
    t = len(driver)
    result = np.zeros((t, 1))
    for i in range(t):
        if (driver[i, ...] != -9999).all():
            result[i, ...] = MODEL(param[i, :], driver[i, ...])
        else:
            result[i, ...] = np.nan
    return result


