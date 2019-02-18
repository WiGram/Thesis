"""
Date:    February 17th, 2019
Authors: Kristian Strand and William Gram
Subject: Applying VAR models to Bloomberg time series

Description:
This script investigates VAR-modelling of Bloomberg
time series using statsmodels' various VAR and
DynamicVAR tools.
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR

mdata = sm.datasets.macrodata.load_pandas().data

# Preparing the index (dates)
dates = mdata[['year','quarter']].astype(int).astype(str)
qrtly = dates['year'] + 'Q' + dates['quarter']

from statsmodels.tsa.base.datetools import dates_from_str
qrtly = dates_from_str(qrtly)

mdata = mdata[['realgdp', 'realcons', 'realinv']]

mdata.index = pd.DatetimeIndex(qrtly)

data = np.log(mdata).diff().dropna()

model = VAR(data)

results = model.fit(2)

results.summary()

results.plot()
plt.show()

results.plot_acorr()
plt.show()

model.select_order(15)

results = model.fit(maxlags = 15, ic = 'aic')

results.summary()

lag_order = results.k_ar

# Forecast: 5 periods
results.forecast(data.values[-lag_order:], 5)

results.plot_forecast(10)
plt.show()