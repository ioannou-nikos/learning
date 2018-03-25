import quandl
import pandas as pd
import os.path
import pickle
import matplotlib.pyplot as plt
from matplotlib import style


def init_vars():
    api_key = open('quandlapikey.txt', 'r').read()
    states_url = 'https://simple.wikipedia.org/wiki/List_of_U.S._states'
    data_file = "fiddy_states.pickle"
    return (api_key, states_url, data_file)


def read_states_abbreviations(url):
    """
    The function reads from wikipedia the US states abbreviations and
    returns a series with them.
    """
    fiddy_states = pd.read_html(url)
    return fiddy_states[0][0][1:]


def retrieve_data_from_url(api_key, states):
    """
    This functions retrieves the housing data from specific url and
    returns them as a pandas DataFrame.
    """
    main_df = pd.DataFrame()
    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        # Get the data from quandl
        df = quandl.get(query, authtoken=api_key)
        df.columns = [abbv]
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
    return main_df


# The main part of the module
if __name__ == "__main__":
    # Check if the datafile exists
    data = None
    # Init the variables
    api_key, states_url, data_file = init_vars()
    if not os.path.isfile(data_file):
        # Retrieve state read_states_abbreviations
        states = read_states_abbreviations(states_url)
        # Retrieve data
        data = retrieve_data_from_url(api_key=api_key, states=states)
        # Open a pickle file
        pickle_out = open(data_file, 'wb')
        # Dump the data
        pickle.dump(data, pickle_out)
        # Close the file
        pickle_out.close()
    else:
        print("READ FILE")
        data = pickle.load(open(data_file, 'rb'))

    print(data.head())
    # Set the matplotlib style
    style.use('fivethirtyeight')
    # Create the plot
    data.plot()
    plt.legend().remove()
    plt.show()

    # Lets correlate
    data_correlation = data.corr()
    print(data_correlation)
    print(data_correlation.describe())
