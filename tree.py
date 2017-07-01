"""
Implements a Tree class to work with Stanford Sentiment Treebank.
"""
import util


class Node:

    def __init__(self, label=None, word=None, parent=None, left=None, right=None, leaf=False, **kwargs):
        self.label = label
        self.word = word
        self.parent = parent
        self.left = left
        self.right = right
        self.is_leaf = leaf

    def df_traverse(self, f=None, args=None):
        f(self, args)
        if self.left is not None:
            self.left.df_traverse(f, args)
        if self.right is not None:
            self.right.df_traverse(f, args)

    def depth(self):
        d = 0
        node = self
        while (node.parent is not None):
            d += 1
            node = node.parent
        return d


class Tree:

    def __init__(self, string, open_char='(', close_char=')', **kwargs):
        self.open_char = open_char
        self.close_char = close_char
        # add spaces before/after open/close chars
        string = string.lower()
        string = string.replace(self.open_char, ' ' + self.open_char + ' ')
        string = string.replace(self.close_char, ' ' + self.close_char + ' ')
        tokens = [t for t in string.strip().split()]
        self.root = self.parse(tokens=tokens)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open_char, "Illegal tree string (open character)"
        assert tokens[-1] == self.close_char, "Illegal tree string (close character)"

        count_open = count_close = 0
        split_idx = 2   # position after the root open_char and label

        # find where left and right child split
        if tokens[split_idx] == self.open_char:
            count_open += 1
            split_idx += 1
        while count_open != count_close:
            if tokens[split_idx] == self.open_char:
                count_open += 1
            if tokens[split_idx] == self.close_char:
                count_close += 1
            split_idx += 1

        node = Node(label=util.int_or_none(tokens[1]), parent=parent)
        if count_open == 0:
            node.word = ''.join(tokens[2:-1])
            node.is_leaf = True
            return node
        node.left = self.parse(tokens=tokens[2:split_idx], parent=node)
        node.right = self.parse(tokens=tokens[split_idx:-1], parent=node)
        return node

    def display(self):
        pass
