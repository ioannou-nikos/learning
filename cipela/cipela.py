#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import the packages needed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create some dummy data to use assuming 20 workers.
# First 20 rows is about their gain/loss next 20 about their actions
workers = np.random.randint(-2, 3, (20, 18))
# Create a dataframe based on data
wdf = pd.DataFrame(workers, columns=['X_eisigitis', 'X_proistamenos',
                                     'X_diefthintis', 'X_genikos',
                                     'X_ektelestikos', 'X_entetalmenos',
                                     'X_topikos', 'X_thematikos', 'X_perifereiarxis',
                                     'Y_eisigitis', 'Y_proistamenos',
                                     'Y_diefthintis', 'Y_genikos',
                                     'Y_ektelestikos', 'Y_entetalmenos',
                                     'Y_topikos', 'Y_thematikos', 'Y_perifereiarxis'])

# From now on start plotting
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(wdf.X_eisigitis)
plt.show()
