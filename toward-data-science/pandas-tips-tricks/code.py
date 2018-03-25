import pandas as pd

df = pd.read_csv("raw-data.csv", sep=',', names=['name','activity','timestamp'])

# Split the name column with UFunc split
df['name'] = df.name.str.split(" ", expand=True)

print(df.name)