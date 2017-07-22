#!/bin/env python3

import argparse

import rntn
import tree as tr

DATA_DIR = "trees"


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dim", type=int, default=25,
                        help="Vector dimension")
    parser.add_argument("-k", "--output-dim", type=int, default=5,
                        help="Number of output classes")
    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("-f", "--dataset", type=str, default="train",
                        choices=['train', 'dev', 'test'], help="Dataset")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument("-b", "--batch-size", type=int, default=30,
                        help="Batch size")
    parser.add_argument("-r", "--reg", type=float, default=1e-6,
                        help="Regularization parameter")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Test a model")
    parser.add_argument("-m", "--model", type=str, default='models/RNTN.pickle',
                        help="Model file")
    args = parser.parse_args()

    # Test
    if args.test:
        print("Testing...")
        model = rntn.RNTN.load(args.model)
        test_trees = tr.load_trees(args.dataset)
        cost, correct, total = model.test(test_trees)
        accuracy = correct * 100.0 / total
        print("Cost = {:.2f}, Correct = {:.0f} / {:.0f}, Accuracy = {:.2f} %".format(
            cost, correct, total, accuracy))
    else:
        # Initialize the model
        model = rntn.RNTN(
            dim=args.dim, output_dim=args.output_dim, batch_size=args.batch_size,
            reg=args.reg, learning_rate=args.learning_rate, max_epochs=args.epochs)

        # Train
        train_trees = tr.load_trees(args.dataset)
        model.fit(train_trees, export_filename=args.model)


if __name__ == '__main__':
    main()
