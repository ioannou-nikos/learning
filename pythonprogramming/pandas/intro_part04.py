# -*- coding:utf-8 -*-

import quandl
import pandas as pd

api_key = open('quandlapikey.txt', 'r').read()

# COMBINING DATAFRAMES
# Concatenation
def combine_dataframes():
    df1 = pd.DataFrame({'HPI': [80, 85, 88, 85],
                        'Int_rate': [2, 3, 2, 2],
                        'US_GDP_Thousands': [50, 55, 65, 55]},
                       index=[2001, 2002, 2003, 2004])
    df2 = pd.DataFrame({'HPI': [80, 85, 88, 85],
                        'Int_rate': [2, 3, 2, 2],
                        'US_GDP_Thousands': [50, 55, 65, 55]},
                       index=[2005, 2006, 2007, 2008])
    df3 = pd.DataFrame({'HPI': [80, 85, 88, 85],
                        'Int_rate': [2, 3, 2, 2],
                        'Low_tier_HPI': [50, 52, 50, 53]},
                       index=[2001, 2002, 2003, 2004])

    # CONATENATION
    # concat = pd.concat([df1, df2])
    # print(concat)
    # concat = pd.concat([df1, df2, df3])
    # print(concat)

    # APPENDING
    df4 = df1.append(df2)
    print(df4)
# df = quandl.get("FMAC/HPI_TX", authtoken=api_key)
# print(df.head())

# Use of pandas read_html in order to get the states abbreviations


fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
# Since read_html returns a list of dataframes, lets take the first one
# print(fiddy_states[0])

# Now iterate through abbreviations skipping first row which is the label
for abbv in fiddy_states[0][0][1:]:
    # print(abbv)
    print("FMAC/HPI_" + str(abbv))

if __name__ == "__main__":
    combine_dataframes()
