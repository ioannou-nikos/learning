import pandas as pd
import matplotlib.pyplot as plt

# Sample financial data for horiszontal analysis
# Assuming this is yearly data for revenue and expenses over a 5-year period
data = {
    'Year':['2016', '2017', '2018', '2019', '2020'],
    'Revenue': [100000,120000,140000,160000,180000],
    'Expenses': [80000, 85000, 90000, 95000, 100000]
}
# Convert the data into a pandas dataframe
df = pd.DataFrame(data=data)

# Set the 'Year' column as the index
df.set_index('Year', inplace=True)

# Perform Horizontal Analysis
# Calculate the change from the base year (2016) for each year as percentage
base_year = df.iloc[0] # first row represents base year
df_horizontal_analysis = (df - base_year)/base_year * 100

# Plotting the results of the horizontal analysis
plt.figure(figsize=(10,6))
for column in df_horizontal_analysis.columns:
    plt.plot(df_horizontal_analysis.index, df_horizontal_analysis[column], marker='o',label=column)
plt.title('Horizontal Analysis of Financial Data')
plt.xlabel('Year')
plt.ylabel('Percentage Change from Base Year(%)')
plt.legend()
plt.grid(True)
plt.show()

# Print the results
print("Results of Horizontal Analysis:")
print(df_horizontal_analysis)