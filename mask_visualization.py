#!/usr/bin/python3

import argparse
import numpy as np
import os
from work_with_data import get_reformated_train_df, show_mask


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=int, default=0, help='int: [1..4]')
    return parser


if __name__ == '__main__':
    parser = createParser()
    args = parser.parse_args()
    train_df = get_reformated_train_df("./input/train.csv")
    img_num = len(os.listdir("./input/train_images"))
    defects_num = len(train_df)
    print("the number of images with no defects: {}".format(img_num - defects_num))
    print("the number of images with defects: {}".format(defects_num))
    if 0 < args.type < 5:
        train_df = train_df[train_df[args.type].notna()]
    images_id = train_df.index
    for image_id in np.random.choice(images_id, 5, replace=False):
        show_mask(train_df, './input/train_images/', image_id)
