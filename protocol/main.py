# -*- coding:utf-8 -*-

"""
Το αρχείο περιλαμβάνει το ουσιαστικό μέρος της προσπάθειας.
"""

import pandas as pd


def read_data_file(skip_footer=0):
    col_names = ['ap', 'user', 'in_subject', 'out_subject', 'in_date',
                 'out_date', 'diff']
    return pd.read_excel("protokola2017.xlsx", skiprows=0, names=col_names,
                         skip_footer=skip_footer)


if __name__ == "__main__":
    read_data_file()
