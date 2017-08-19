# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

# Generate a noisy line, and save the data in a pandas-DataFrame
x = np.arange(100)
y = 0.5*x - 20 + np.random.randn(len(x))
df = pd.DataFrame({'x': x, 'y': y})

# Fit a linear model, using the "formula" language
# added by the package "patsy"
model = sm.ols('y~x', data=df).fit()
print(model.summary())
