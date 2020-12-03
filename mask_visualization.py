#!/usr/bin/python3

import argparse
import numpy as np
from work_with_data import get_reformated_train_df, extend_train_df, show_mask


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t', '--type', type=int, default=0, help='int: [1..4]')
    argparser.add_argument('-n', '--number', type=int, default=5, help='number of displayed images')
    return argparser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    train_df = get_reformated_train_df('./input/train.csv')
    defects_num = len(train_df)
    train_df = extend_train_df(train_df, './input/train_images')
    img_num = len(train_df)

    print('the number of images with no defects: {}'.format(img_num - defects_num))
    print('the number of images with defects: {}'.format(defects_num))
    if 0 < args.type < 5:
        train_df = train_df[train_df[args.type].notna()]
    images_id = train_df.index
    for image_id in np.random.choice(images_id, args.number, replace=False):
        show_mask(train_df, './input/train_images/', image_id)
