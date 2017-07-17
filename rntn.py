"""
Implementation of the Recursive Neural Tensor Network (RNTN) model
"""

import collections
import pickle
import time

import numpy as np

import tree as tr


class RNTN:

    def __init__(self, dim=10, output_dim=5, batch_size=30, reg=10, learning_rate=1e-2, max_epochs=2):
        self.dim = dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.reg = reg
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def fit(self, trees, export_filename='models/RNTN.pickle'):
        import sgd

        self.word_map = tr.load_word_map()
        self.num_words = len(self.word_map)
        self.init_params()
        self.optimizer = sgd.SGD(self, self.learning_rate, self.batch_size)

        for epoch in range(self.max_epochs):
            print("Running epoch {} ...".format(epoch))
            start = time.time()
            self.optimizer.optimize(trees)
            end = time.time()
            print("   Time per epoch = {:.4f}".format(end-start))

            # Save the model
            self.save(export_filename)

    def test(self, trees):
        raise NotImplementedError

    def predict(self, tree):
        if tr.isleaf(tree):
            # output = word vector
            try:
                tree.vector = self.L[:, self.word_map[tree[0]]]
            except:
                tree.vector = self.L[:, self.word_map[tr.UNK]]
        else:
            # calculate output of child nodes
            self.predict(tree[0])
            self.predict(tree[1])

            # compute output
            lr = np.hstack([tree[0].vector, tree[1].vector])
            tree.vector = np.tanh(
                np.tensordot(self.V, np.outer(lr, lr), axes=([1, 2], [0, 1])) +
                np.dot(self.W, lr) + self.b)

        # softmax
        import util
        tree.output = util.softmax(np.dot(self.Ws, tree.vector) + self.bs)
        label = np.argmax(tree.output)
        tree.set_label(str(label))
        return tree

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.dim, f)
            pickle.dump(self.output_dim, f)
            pickle.dump(self.batch_size, f)
            pickle.dump(self.reg, f)
            pickle.dump(self.learning_rate, f)
            pickle.dump(self.max_epochs, f)
            pickle.dump(self.stack, f)
            # pickle.dump(self.word_map, f)

    def load(filename):
        with open(filename, 'rb') as f:
            dim = pickle.load(f)
            output_dim = pickle.load(f)
            batch_size = pickle.load(f)
            reg = pickle.load(f)
            learning_rate = pickle.load(f)
            max_epochs = pickle.load(f)
            stack = pickle.load(f)
            model = RNTN(dim=dim, output_dim=output_dim, batch_size=batch_size,
                         reg=reg, learning_rate=learning_rate, max_epochs=max_epochs)
            model.stack = stack
            model.L, model.V, model.W, model.b, model.Ws, model.bs = model.stack
            # model.word_map = pickle.load(f)
            return model

    def init_params(self):
        print("Initializing RNTN parameters...")

        # word vectors
        self.L = 0.01 * np.random.randn(self.dim, self.num_words)

        # RNTN parameters
        self.V = 0.01 * np.random.randn(self.dim, 2*self.dim, 2*self.dim)
        self.W = 0.01 * np.random.randn(self.dim, 2*self.dim)
        self.b = 0.01 * np.random.randn(self.dim)

        # Softmax parameters
        self.Ws = 0.01 * np.random.randn(self.output_dim, self.dim)
        self.bs = 0.01 * np.random.randn(self.output_dim)

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty_like(self.V)
        self.dW = np.empty_like(self.W)
        self.db = np.empty_like(self.b)
        self.dWs = np.empty_like(self.Ws)
        self.dbs = np.empty_like(self.bs)

    def cost_and_grad(self, trees, test=False):
        cost, correct, total = 0.0, 0.0, 0.0
        self.L, self.V, self.W, self.b, self.Ws, self.bs = self.stack

        # Initialize gradients
        self.dL = collections.defaultdict(lambda: np.zeros((self.dim,)))
        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0

        # Forward propagation
        for tree in trees:
            _cost, _correct, _total = self.forward_prop(tree)
            cost += _cost
            correct += _correct
            total += _total

        if test:
            return cost / len(trees), correct, total

        # Back propagattion
        for tree in trees:
            self.back_prop(tree)

        # Scale cost and gradients by minibatch size
        scale = 1.0 / self.batch_size
        for v in self.dL.values():
            v *= scale

        # Add L2 reguralization
        cost += 0.5 * self.reg * np.sum(self.V ** 2)
        cost += 0.5 * self.reg * np.sum(self.W ** 2)
        cost += 0.5 * self.reg * np.sum(self.Ws ** 2)
        cost *= scale

        grad = [self.dL,
                scale * (self.dV + (self.reg * self.V)),
                scale * (self.dW + (self.reg * self.W)),
                scale * self.db,
                scale * (self.dWs + (self.reg * self.Ws)),
                scale * self.dbs]

        return cost, grad

    def forward_prop(self, tree):
        cost, correct, total = 0.0, 0.0, 0.0

        if tr.isleaf(tree):
            # output = word vector
            tree.vector = self.L[:, self.word_map[tree[0]]]
        else:
            # calculate output of child nodes
            lcost, lcorrect, ltotal = self.forward_prop(tree[0])
            rcost, rcorrect, rtotal = self.forward_prop(tree[1])
            cost += lcost + rcost
            correct += lcorrect + rcorrect
            total += ltotal + rtotal

            # compute output
            lr = np.hstack([tree[0].vector, tree[1].vector])
            tree.vector = np.tanh(
                np.tensordot(self.V, np.outer(lr, lr), axes=([1, 2], [0, 1])) +
                np.dot(self.W, lr) + self.b)

        # softmax
        import util
        tree.output = util.softmax(np.dot(self.Ws, tree.vector) + self.bs)

        # cost
        cost -= np.log(tree.output[int(tree.label())])
        correct += (np.argmax(tree.output) == int(tree.label()))
        total += 1

        return cost, correct, total

    def back_prop(self, tree, error=None):
        # softmax grad
        deltas = tree.output
        deltas[int(tree.label())] -= 1.0
        self.dWs += np.outer(deltas, tree.vector)
        self.dbs += deltas
        deltas = np.dot(self.Ws.T, deltas)
        if error is not None:
            deltas += error
        deltas *= (1 - tree.vector**2)

        # leaf node => update word vectors
        if tr.isleaf(tree):
            self.dL[self.word_map[tree[0]]] += deltas
            return

        # Hidden gradients
        else:
            lr = np.hstack([tree[0].vector, tree[1].vector])
            outer = np.outer(deltas, lr)
            self.dV += (np.outer(lr, lr)[:, :, None] * deltas).T
            self.dW += outer
            self.db += deltas

            # Compute error for children
            deltas = np.dot(self.W.T, deltas)
            deltas += np.tensordot(self.V.transpose((0,2,1)) + self.V, outer.T,
                                   axes=([1,0], [0,1]))

            self.back_prop(tree[0], deltas[:self.dim])
            self.back_prop(tree[1], deltas[self.dim:])

    def update_params(self, scale, update):
        self.stack[1:] = [P+scale*dP for P, dP in zip(self.stack[1:], update[1:])]
        # Update L separately
        dL = update[0]
        for j in dL.keys():
            self.L[:,j] += scale*dL[j]