"""
Implements a Tree class to work with Stanford Sentiment Treebank.
"""
import util

UNK = 'UNK'     # denotes all unknown words


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

    def tree_string(self):
        if self.is_leaf:
            return "({} {})".format(self.label, self.word)
        else:
            return "({} {} {})".format(
                self.label,
                self.left.tree_string(),
                self.right.tree_string(),
            )

    def formatted_string(self, depth=0):
        indentation = '    ' * depth
        s = indentation
        if self.is_leaf:
            s += "({} {})".format(self.label, self.word)
        else:
            s += "({}".format(self.label)
            s += "\n" + self.left.formatted_string(depth + 1)
            s += "\n" + self.right.formatted_string(depth + 1)
            s += "\n" + indentation + ")"
        return s


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

    def __str__(self):
        return self.root.tree_string()

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
        print(self.root.formatted_string())


if __name__ == '__main__':
    tree_string = "(5 (3 I) (5 (5 love) (3 it)))"
    tree = Tree(tree_string)
    tree.display()
