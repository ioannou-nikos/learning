import pandas as pd
import matplotlib.pyplot as plt

# Sample financial data for trend analysis
# Lets assume this is yearly revenue data for a company over 5-year period
data = {
    'Year':['2016', '2017', '2018', '2019', '2020'],
    'Revenue': [100000,120000,140000,160000,180000],
    'Expenses': [80000, 85000, 90000, 95000, 100000]
}

# Convert the data into a pandas dataframe
df = pd.DataFrame(data=data)

# Set the 'Year' column as the index
df.set_index('Year', inplace=True)

# Calculate the Year-over-Year (YoY) growth for Revenue as Expenses
df['Revenue Growth'] = df['Revenue'].pct_change() * 100
df['Expenses Growth'] = df['Expenses'].pct_change() * 100

# Plotting the trend analysis
plt.figure(figsize=(10,5))
# Plot Revenue and Expenses over time
plt.subplot(1,2,1)
plt.plot(df.index, df['Revenue'], marker='o', label='Revenue')
plt.plot(df.index, df['Expenses'], marker='o', linestyle='--',label='Expenses')
plt.title('Revenue and Expenses Over Time')
plt.xlabel('Year')
plt.ylabel('Amount ($)')
plt.legend()

# Plot Growth over time
plt.subplot(1,2,2)
plt.plot(df.index, df['Revenue Growth'], marker='o', label='Revenue Growth')
plt.plot(df.index, df['Expenses Growth'], marker='o', linestyle='--',label='Expenses Growth')
plt.title('Growth Year-over-Year')
plt.xlabel('Year')
plt.ylabel('Growth (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Displaying growth rates
print("Year-over-Year Growth Rates:")
print(df[['Revenue Growth', 'Expenses Growth']])