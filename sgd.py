#!/bin/env python3

import random
import time


class SGD:

    def __init__(self, model, learning_rate=1e-2, batch_size=30):
        self.model = model
        assert self.model is not None, "Please provide a model to optimize!"

        self.iter = 0
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def optimize(self, trees, log_interval=1):
        m = len(trees)

        # Randomly shuffle data
        random.shuffle(trees)

        self.iter = 0
        for i in range(0, 1 + m - self.batch_size, self.batch_size):
            start = time.time()
            self.iter += 1
            data = trees[i: i+self.batch_size]
            cost, grad = self.model.cost_and_grad(data)

            # Perform one step of parameter update
            scale = -self.learning_rate
            update = grad
            self.model.update_params(scale=scale, update=update)
            end = time.time()

            # Log status
            if self.iter % log_interval == 0:
                print("\r   Iter = {}, Cost = {:.4f}, Time = {:.4f}".format(self.iter, cost,
                                                                            end-start),
                      end=' ')
