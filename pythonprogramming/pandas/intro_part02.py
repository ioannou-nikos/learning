# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

web_stats = {'Day': [1, 2, 3, 4, 5, 6],
             'Visitors': [43, 34, 65, 56, 29, 76],
             'Bounce Rate': [65, 67, 78, 65, 45, 52]}
df = pd.DataFrame(web_stats)
# Print first rows
print(df.head())
# Print last 2 rows
print(df.tail(2))
# Now set the Day column to be the index and do it inplace
df.set_index('Day', inplace=True)
print(df.head())

# Now for some plotting
style.use('fivethirtyeight')
df['Visitors'].plot()
plt.show()

df.plot()
plt.show()
