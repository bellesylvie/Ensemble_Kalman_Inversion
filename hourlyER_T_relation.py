import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import least_squares
import numpy as np


def ER(T, p):
    # estimate the hourly ER with air or soil temperature
    T0 = -46.02
    Tref = 15
    Rb, E0 = p
    r = Rb * np.exp(E0 * (1 / (Tref - T0) - 1 / (T - T0)))
    return r


def residuals(p, y, x):
    return y-ER(x, p)


ER2016 = pd.data = pd.read_excel(r'G:\论文\生态\生态系统呼吸昼夜变化\Järveoja et al. data sets.xlsx', sheet_name='2016 AC ER')
ER2016['datetime'] = pd.to_datetime(ER2016[['year', 'month', 'day', 'hour']])



# using the data shared by Järveoja et al. (2020) to see the relationship between ecosystem respiration and temperature
# at hourly scale in a northern peatland ecosystem.

# ER2015 = pd.data = pd.read_csv(r'G:\论文\生态\生态系统呼吸昼夜变化\Järveoja et al. data sets-绘制每天ER变化曲线图.csv')
# ER2015['datetime'] = pd.to_datetime(ER2015[['year', 'month', 'day', 'hour']])
# ER2015.set_index('datetime', inplace=True)
#
#
# y = np.array(ER2015.loc['20150801', 'flux'].values)
#
# p0 = [17, 309]
# Ta = np.array(ER2015.loc['20150801', 'CO2_Ta_amb'].values)
# model_Ta = least_squares(residuals, p0, args=(y, Ta), bounds=([0, 0], [200, 500]))
# y_fitted_Ta = func(Ta, model_Ta['x'])
# print('Ta拟合参数', model_Ta['x'])
#
# Ts = np.array(ER2015.loc['20150801', 'Ts10'].values)
# model_Ts = least_squares(residuals, p0, args=(y, Ts), bounds=([0, 0], [200, 500]))
# y_fitted_Ts = func(Ts, model_Ts['x'])
# print('Ts拟合参数', model_Ts['x'])
#
# fig = plt.figure()
# host = fig.add_subplot(1, 1, 1)
# ax2 = host.twinx()
# ax3 = host.twinx()
# host.set_xlabel("Time")
# host.set_ylabel("observed ER")
# ax2.set_ylabel("ER estimated with Ts")
# ax3.set_ylabel("ER estimated with Ta")
#
# color1, color2, color3 = plt.cm.rainbow([0, .5, .9])
# p1 = host.plot(ER2015.loc['20150801'].index, ER2015.loc['20150801', 'flux'], color=color1,
#                label="ER measured by opaque automated chamber")
# p2 = ax2.plot(ER2015.loc['20150801'].index, y_fitted_Ts, color=color2,
#               label="Soil temperature at 10 cm depth")
# p3 = ax3.plot(ER2015.loc['20150801'].index, y_fitted_Ta, color=color3, label="Air temperature")
# host.legend(handles=p1 + p2 + p3, loc='best')
# ax3.spines['right'].set_position(('outward', 60))
# plt.show()


