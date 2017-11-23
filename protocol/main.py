# -*- coding:utf-8 -*-

"""
Το αρχείο περιλαμβάνει το ουσιαστικό μέρος της προσπάθειας.
"""

import pandas as pd


def get_enotita_code(text):
    """
    Στη συνάρτηση αυτή από ένα κείμενο επιστρέφουμε τον κωδικό της ενότητας.
    """
    key_words = ["μαθία", "εσσαλονί", "ιλκίς", "έλλας", "ιερίας", "ερρών",
                 "αλκιδικής"]
    for en in key_words:
        if en in text:
            return key_words.index(en)
    return -1


def get_enotita_codes_from_series(user_col):
    """
    Πάρε τη στήλη user και δημιούργησε μια σειρά από κωδικούς
    """
    codes = [get_enotita_code(usr) for usr in user_col]
    return codes


def read_data_file(skip_footer=0):
    col_names = ['ap', 'user', 'in_subject', 'out_subject', 'in_date',
                 'out_date', 'diff']
    return pd.read_excel("protokola2017.xlsx", skiprows=0, names=col_names,
                         skip_footer=skip_footer)


if __name__ == "__main__":
    read_data_file()
