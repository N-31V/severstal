#!/usr/bin/python3

import argparse
from enet import ENet
from work_with_model import ModelToolkit, load_model


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epochs', type=int, default=20, help='int')
    argparser.add_argument('-n', '--num_workers', type=int, default=4, help='int')
    argparser.add_argument('-b', '--batch_size', type=int, default=4, help='int')
    argparser.add_argument('-m', '--model', type=str, default=None, help='model path')
    return argparser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    if args.model is None:
        model = ENet(4)
        model = ModelToolkit(model, 'ENet')
    else:
        model = load_model(args.model)
    model.train(args.epochs, args.batch_size, args.num_workers)
