#!/usr/bin/python3

import argparse
from work_with_model import ModelToolkit, load_model


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--model', type=str, required=True, help='model path')
    argparser.add_argument('-n', '--num_workers', type=int, default=4, help='int')
    argparser.add_argument('-b', '--batch_size', type=int, default=4, help='int')
    return argparser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    model = load_model(args.model)
    pred_df = model.predict(args.batch_size, args.num_workers)
    pred_df.to_csv('test.csv')

