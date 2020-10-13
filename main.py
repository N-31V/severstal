#!/usr/bin/python3

import segmentation_models_pytorch as smp
from training import Trainer
from other_func import plot_score



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
    model_trainer = Trainer(model, "./input/", "./input/train.csv", num_epochs=args.epochs, num_workers=args.num_workers, batcg_size=args.batch_size)
    model_trainer.start()

    plot_score(model_trainer.losses, "BCE loss")
    plot_score(model_trainer.dice_scores, "Dice score")
    plot_score(model_trainer.iou_scores, "IoU score")
