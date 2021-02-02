#!/usr/bin/python3

import argparse
import segmentation_models_pytorch as smp
from work_with_model import ModelToolkit, load_model


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--model', type=str, required=True, help='model path')
    return argparser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    model = load_model(args.model)
    model.plot_scores()
