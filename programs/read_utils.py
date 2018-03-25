# -*- coding:utf-8 -*-

"""
Author: Nikos Ioannou
This file is for holding functions to read the files and work with each line
"""

import pandas as pd


def read_line(row, col_count):
    """
    :param row: The row to work with each line and confirm that is valid or not
    :param col_count: The number of columns that a line must have
    :return:
    """



def read_file(f, sheet_loc=0, col_count=1):
    """

    :param f: The file to read
    :param sheet_loc: The int representing the sheet number (0 based)
    :param col_count: The number of columns that is required
    :return:
    """
    df = pd.read_excel(f, sheet_loc)
    for index, row in df.iterrows():
        if read_line(row,col_count=col_count):
            print("Line is valid")
        else:
            print("Line is invalid")