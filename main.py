#!/usr/bin/python3

import argparse
import segmentation_models_pytorch as smp
from work_with_model import ModelToolkit


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=20, help='int')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='int')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='int')
    return parser


if __name__ == '__main__':
    parser = createParser()
    args = parser.parse_args()
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
    model = ModelToolkit(model, 'Unet', num_workers=args.num_workers, batch_size=args.batch_size)
    model.train(args.epochs)
