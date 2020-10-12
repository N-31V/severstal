#!/usr/bin/python3

import segmentation_models_pytorch as smp
from training import Trainer
from other_func import plot_score

if __name__ == '__main__':
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
    model_trainer = Trainer(model, "./input/", "./input/train.csv")
    model_trainer.start()

    plot_score(model_trainer.losses, "BCE loss")
    plot_score(model_trainer.dice_scores, "Dice score")
    plot_score(model_trainer.iou_scores, "IoU score")
