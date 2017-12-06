# -*- coding:utf-8 -*-

import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2015, 8, 22)

df = web.DataReader("XOM", "yahoo", start, end)
print(df.head())

# Now for some visualization
style.use('fivethirtyeight')
df['High'].plot()
plt.legend()
plt.show()
