# -*- coding: utf-8 -*-


def top_counts(count_dict, n=10):
    """
    Compute the n top pairs of the dictionary holding the keys with their counts
    :param count_dict: The dictionary with the keys and corresponding counts
    :param n: The count of top values
    :return: The key, count pairs
    """
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]
