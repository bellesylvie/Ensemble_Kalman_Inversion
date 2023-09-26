# Ensemble_Kalman_Inversion

In our case to estimate the time series of the parameters for the model proposed by lasslop et al., 2010, EnKF was used as a way to iteratively estimate parameters. 

In this case, we put both the parameters of the model(i.e., alpha, beta0, Rref, E0) and the observation variable NEE into the state vector (i.e., [alpha, beta0,Rref, E0, NEE]) of the EnKF. We create an artificial dynamical system by assuming that the parameters at current time equal the parameters at previous time, and the observation operator H is [0, I].
