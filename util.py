"""
Utility functions
"""


def int_or_none(string):
    try:
        return int(string)
    except ValueError:
        return None
