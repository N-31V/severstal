#!/usr/bin/python3
import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from other_func import show_mask


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=int, default=3, help='int: [1..4]')
    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args()
    train_df = pd.read_csv("./input/train.csv")
    img_num = len(os.listdir("./input/train_images"))
    defects_num = len(train_df.drop_duplicates(subset=['ImageId']))
    print("the number of images with no defects: {}".format(img_num - defects_num))
    print("the number of images with defects: {}".format(defects_num))
    y = []
    for i in range(4):
        n = len(train_df[train_df['ClassId'] == (i + 1)])
        print("the number of images with type {} defects: {}".format(i + 1, n))
        y.append(n)

    temp = list(train_df['ImageId'].value_counts().to_frame()['ImageId'].value_counts())
    temp.insert(0, img_num - defects_num)
    for i, count in enumerate(temp):
        print("{} number of classes in {} images".format(i, count))

    names = train_df.drop_duplicates(subset=['ImageId']).loc[
        train_df['ClassId'] == namespace.type, 'ImageId'].to_numpy()
    for name in np.random.choice(names, 5, replace=False):
        show_mask(train_df, './input/train_images/', name)
