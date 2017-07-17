#!/bin/env python3

import unittest

from nltk.tree import ParentedTree


class Test(unittest.TestCase):

    def test(self):
        filename = 'trees/dev.txt'
        print("Reading dev.txt")
        with open(filename) as f:
            trees = [ParentedTree.fromstring(s) for s in f]

        tree = trees[0]
        print(type(tree.label()))
        print(tree)


if __name__ == '__main__':
    unittest.main()
