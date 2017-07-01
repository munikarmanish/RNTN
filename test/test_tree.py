"""
Unit tests for the Tree class.
"""

import unittest

from tree import Tree


class TreeTest(unittest.TestCase):

    def setUp(self):
        self.tree_string = "(5 (3 I) (5 (5 love) (3 it)))"
        self.tree = Tree(self.tree_string)
        self.root = self.tree.root
        self.i = self.root.left
        self.love_it = self.root.right
        self.love = self.love_it.left
        self.it = self.love_it.right

    def test_parse(self):
        self.assertEqual(5, self.root.label)
        self.assertEqual(3, self.i.label)
        self.assertEqual(5, self.love_it.label)
        self.assertEqual(5, self.love.label)
        self.assertEqual(3, self.it.label)

        # self.assertEqual("i love it", self.root.word)
        self.assertEqual("i", self.i.word)
        # self.assertEqual("love it", self.love_it.word)
        self.assertEqual("love", self.love.word)
        self.assertEqual("it", self.it.word)

    def test_depth(self):
        self.assertEqual(0, self.root.depth())
        self.assertEqual(1, self.i.depth())
        self.assertEqual(1, self.love_it.depth())
        self.assertEqual(2, self.love.depth())
        self.assertEqual(2, self.it.depth())

    def test_tree_string(self):
        self.assertEqual(self.tree_string.lower(), str(self.tree))
