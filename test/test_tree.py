"""
Unit tests for the Tree class.
"""

import unittest

from tree import Tree


class TreeTest(unittest.TestCase):

    def test_parse(self):
        tree_string = "(5 (3 I) (5 (5 love) (3 it)))"

        root = Tree(tree_string).root
        i = root.left
        love_it = root.right
        love = love_it.left
        it = love_it.right

        self.assertEqual(5, root.label)
        self.assertEqual(3, i.label)
        self.assertEqual(5, love_it.label)
        self.assertEqual(5, love.label)
        self.assertEqual(3, it.label)

        # self.assertEqual("i love it", root.word)
        self.assertEqual("i", i.word)
        # self.assertEqual("love it", love_it.word)
        self.assertEqual("love", love.word)
        self.assertEqual("it", it.word)
