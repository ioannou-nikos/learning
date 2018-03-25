import quandl
import pands as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

api_key = open('quandlapikey.txt', 'r').read()


def state_list():
    state_url = 'https://simple.wikipedia.org/wiki/List_of_U.S._states'
    fiddy_states = pd.read_html(state_url)
    return fiddy_states[0][0][1:]

def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = quandl.get(query, authtoken=api_key)
        print(query)
        df[abbv] = (df[abbv]-df[abbv][0]) / df[abbv][0] * 100.0
        print(df.head())
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

        pickle_out = open('fiddy_states3.pickle', 'wb')
        pickle.dump(main_df, pickle_out)
        pickle_out.close()


def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df["United States"] = (df["United States"] - df["United States"][0]) / df["United States"][0] * 100.0
    return df


fig = plt.figure()
ax1 = plt.subplot2grid((1, 1), (0, 0))
HPI_data = pd.read_pickle('fiddy_states3.pickle')
HPI_State_Correlation = HPI_data.corr()
