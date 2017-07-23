#!/bin/env python3

import unittest

import tree as tr


def f(model, text):
    """
    This function prints the predicted sentiment labeled parse tree.

    NOTE: It requires CoreNLP server running at http://localhost:9000.

    Parameters
    ----------
    model := the RNTN model
    text := a sentence to predict

    Examples
    --------
    >>> model = rntn.RNTN.load('models/RNTN.pickle')
    >>> f(model, "not very good")
           1
      -----|-----
      |         4
      |      ---|---
      2      2     3
      |      |     |
     not   very   good

    """
    for tree in tr.parse(text):
        model.predict(tree).pretty_print()


if __name__ == '__main__':
    unittest.main()
