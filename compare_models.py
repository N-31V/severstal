#!/usr/bin/python3

import argparse
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from work_with_model import ModelToolkit, load_model


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--models_folder', type=str, required=True, help='model path')
    return argparser


def compare_models(names, scores, func_name):
    for name, score in zip(names, scores):
        plt.plot(score, label=name)
    plt.title(f'{func_name} plot')
    plt.xlabel('Epoch')
    plt.ylabel(f'{func_name}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    models = os.listdir(args.models_folder)
    names = []
    train_losses = []
    val_losses = []
    train_dice = []
    val_dice = []
    for m in models:
        model = load_model(os.path.join(args.models_folder, m))
        names.append(model.name)
        train_losses.append(model.losses['train'])
        val_losses.append(model.losses['val'])
        train_dice.append(model.scores['train'].dice_scores)
        val_dice.append(model.scores['val'].dice_scores)
        print(model.name, model.scores['val'].dice_scores[-1])

    compare_models(names, train_losses, 'train loss')
    compare_models(names, val_losses, 'val loss')
    compare_models(names, train_dice, 'train dice')
    compare_models(names, val_dice, 'val dice')


