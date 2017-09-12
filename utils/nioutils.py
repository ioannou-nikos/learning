# -*- coding:utf-8 -*-

"""
Author: Nikos Ioannou

The file contains functions that are of great use throught my work
"""

from difflib import SequenceMatcher  # Used in similar

def clean_greek_letters(phrase):
    """
    Return the letter without any punctuation
    """
    alphabet = {'α':'Α', 'ά':'Α', 'Α':'Α', 'Ά':'Α',
                'β':'Β', 'Β':'Β',
                'γ':'Γ', 'Γ':'Γ',
                'δ':'Δ', 'Δ':'Δ',
                'ε':'Ε', 'έ':'Ε', 'Ε':'Ε', 'Έ':'Ε',
                'ζ':'Ζ', 'Ζ':'Ζ',
                'η':'Η', 'ή':'Η', 'Η':'Η', 'Ή':'Η',
                'θ':'Θ', 'Θ':'Θ',
                'ι':'Ι', 'ί':'Ι', 'ϊ':'Ι', 'Ϊ':'Ι', 'ΐ':'Ι', 'Ι':'Ι',
                'κ':'Κ', 'Κ':'Κ',
                'λ':'Λ', 'Λ':'Λ',
                'μ':'Μ', 'Μ':'Μ',
                'ν':'Ν', 'Ν':'Ν',
                'ξ':'Ξ', 'Ξ':'Ξ',
                'ο':'Ο', 'Ο':'Ο', 'ό':'Ο', 'Ό':'Ο',
                'π':'Π', 'Π':'Π',
                'ρ':'Ρ', 'Ρ':'Ρ',
                'σ':'Σ', 'Σ':'Σ', 'ς':'Σ',
                'τ':'Τ', 'Τ':'Τ',
                'υ':'Υ', 'ύ':'Υ', 'Υ':'Υ', 'Ύ':'Υ',
                'φ':'Φ', 'Φ':'Φ',
                'χ':'Χ', 'Χ':'Χ',
                'ω':'Ω', 'ώ':'Ω', 'Ω':'Ω', 'Ώ':'Ω'}
    if phrase in alphabet:
        return alphabet.get(phrase)
    else:
        return phrase

def clean_greek_phrase(phrase):
    """
    Return the phrase without any punctuation
    """
    l = list(phrase)
    for ind, val in enumerate(l):
        l[ind] = clean_greek_letters(val)
    return "".join(l)

def similar_difflib(str1, str2):
    """
    Return the similarity ratio of two strings
    """
    return SequenceMatcher(None, str1, str2).ratio()


if __name__ == '__main__':
    s = "Βρε καλώς τα παληκάρια"
    print(s)
    print(clean_greek_phrase(s))
    print("Checking Similarity with Greek")
    print(similar_difflib("ΝΙΚΟΣ ΙΩΑΝΝΟΥ", "ΝΙΚΟΛΑΟΣ ΙΩΑΝΝΟΥ"))

