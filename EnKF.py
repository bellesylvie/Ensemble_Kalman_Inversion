import math
import numpy as np
from numpy.linalg import inv
from numpy import dot
import scipy.stats as st


def EKI(Z, H, N, dim_param, dim_obs, forcing, param_ens, obs_noise):
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

    T = len(forcing)
    model = np.zeros((T, dim_param+dim_obs))
    anlys = np.zeros((T, dim_param+dim_obs))
    anlys_std = np.zeros((T, dim_param+dim_obs))
    # low_ci_bounds = np.zeros((T, dim_state))
    # high_ci_bounds = np.zeros((T, dim_state))

    for i in range(T):
        if ~np.isnan(Z[i]):
            # prediction
            V_hat = forecast(forcing[i, :], N, param_ens, dim_param, dim_obs)
            model[i, :] = V_hat.mean(axis=1)
            C_hat = get_C_hat(V_hat, N, dim_param+dim_obs)
            # update
            R = np.eye(dim_obs) * obs_noise
            Z_ens = generate_obs_ensemble(Z[i], R, N, dim_obs)
            update = analysis(V_hat, C_hat, H, Z_ens, N, dim_param, dim_obs, R)

            # record target variables
            anlys[i, :] = update.mean(axis=1)
            anlys_std[i, :] = update.std(axis=1)

            # low_ci_bound, high_ci_bound = st.t.interval(0.95, N - 1, loc=update.mean(axis=1), scale=st.sem(update, axis=1))
            # low_ci_bounds[i, :] = low_ci_bound
            # high_ci_bounds[i, :] = high_ci_bound
            param_ens = update[:dim_param, :]

    return model, anlys, anlys_std


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


def forecast(forcing, N, param_ens, dim_param, dim_obs, var_inflation):
    '''
    This function takes the posterior ensemble from the last step V_0, the parameter dimension dim_param,
    the number of ensemble members N, the forcing data for the NEE model forcing, the dimension of observation dim_obs,
    and whether to do variance inflation var_inflation
    '''

    forecast = np.zeros((dim_param+dim_obs, N))

    if ~var_inflation:
        param_ens_forecast = param_ens
    else:
        param_cov_infl = (1/0.9975-1)*np.diag(np.diag(np.cov(param_ens)))
        param_ens_forecast = np.zeros_like(param_ens)
        for i in range(N):
             param_ens_forecast[:, i] = param_ens[:, i] + np.random.multivariate_normal(np.zeros(dim_param), param_cov_infl, 1)

    forecast[:dim_param, :] = param_ens_forecast
    # calculate NEE
    for i in range(N):
        forecast[-dim_obs, i] = nee_model(param_ens_forecast[:, i], forcing[:])
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


def analysis(V_hat, C, H, Z_ens, N, dim_param, dim_obs, R):
    '''
    This function takes the forecast ensemble V_hat, the forecast covariance matrix C, the observation matrix H,
    the ensemble of the observation Z_ens, the number of ensemble members N, the parameter dimension dim_param, and the
    number of observed components dim_obs.
    It returns the posterior ensemble V.
    '''
    V = np.zeros((dim_param+dim_obs, N))
    K = dot(dot(C, np.transpose(H)), inv(R + dot(H, dot(C, np.transpose(H)))))
    for k in range(N):
        V[:, k] = V_hat[:, k] - dot(K, dot(H, V_hat[:, k]) - Z_ens[:, k])
    return V


def get_C_hat(V_hat, N, d):
    '''
    This function takes the forecast ensemble V_hat, the number of ensemble members N, and the dimension of state vector d.
    It returns the forecast covariance matrix C_hat.
    '''
    V_bar_hat = 1 / N * np.sum(V_hat, axis=1)
    C_hat = np.zeros((d, d))

    for i in range(N):
        temp = V_hat[:, i] - V_bar_hat
        C_hat = C_hat + np.outer(temp, temp)

    C_hat = 1 / (N - 1) * C_hat

    return C_hat


def validation(param, driver):
    t = len(driver)
    result = np.zeros((t, 1))
    for i in range(t):
        if (driver[i, :] != -9999).all():
            result[i, :] = nee_model(param[i, :], driver[i, :])
        else:
            result[i, :] = np.nan
    return result


