#!/usr/bin/python3

import argparse
import segmentation_models_pytorch as smp
from datetime import datetime
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
        model = smp.Unet('resnet18', encoder_weights='imagenet', classes=4, activation=None)
        model = ModelToolkit(model, 'Unet(ResNet18)-created:' + datetime.today().strftime("%d-%m-%Y"),
                             num_workers=args.num_workers, batch_size=args.batch_size)
    else:
        model = load_model(args.model)
    model.train(args.epochs)
