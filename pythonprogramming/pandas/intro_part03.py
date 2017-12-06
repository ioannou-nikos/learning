import pandas as pd

df = pd.read_csv('ZILLOW-Z77006_ZRIFAH.csv')
print(df.head())

# Set an set_index
df.set_index('Date', inplace=True)

# Now save it back to csv
df.to_csv('newcsv2.csv')

# Now read back the csv
df = pd.read_csv('newcsv2.csv')
print(df.head())

# Now read the csv and set the column index at the same time
df = pd.read_csv('newcsv2.csv', index_col=0)
print(df.head())
