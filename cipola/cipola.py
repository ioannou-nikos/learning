#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import the packages needed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Make a function that takes an axe and forms it
def make_axe(ax):
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.plot((-2, 2), (-2, 2))
    ax.plot((-2, 2), (2, -2))
    return ax


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
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(wdf.X_eisigitis, wdf.Y_perifereiarxis)
#ax = make_axe(ax)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

axs[0, 0].plot(wdf.X_eisigitis, wdf.Y_proistamenos, 'bo')
axs[0,0] = make_axe(axs[0,0])


axs[0, 1].scatter(wdf.X_eisigitis, wdf.Y_diefthintis)
axs[0,1] = make_axe(axs[0,1])

axs[1, 0].scatter(wdf.X_eisigitis, wdf.Y_genikos)
axs[1,0] = make_axe(axs[1,0])

axs[1, 1].scatter(wdf.X_eisigitis, wdf.Y_ektelestikos)
axs[1,1] = make_axe(axs[1,1])

"""
axs[2, 0].scatter(wdf.X_eisigitis, wdf.Y_entetalmenos)
axs[2,0] = make_axe(axs[2,0])

axs[2, 1].scatter(wdf.X_eisigitis, wdf.Y_topikos)
axs[2,1] = make_axe(axs[2,1])

axs[3, 0].scatter(wdf.X_eisigitis, wdf.Y_thematikos)
axs[3,0] = make_axe(axs[3,0])

axs[3, 1].scatter(wdf.X_eisigitis, wdf.Y_perifereiarxis)
axs[3,1] = make_axe(axs[3,1])
"""
plt.show()

dd = wdf.groupby([wdf.X_eisigitis, wdf.Y_diefthintis]).count()
print(dd)
