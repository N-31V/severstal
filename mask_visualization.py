#!/usr/bin/python3

import argparse
import numpy as np
from work_with_data import get_reformated_df, extend_df, show_mask


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--csv', required=True, type=str, help='csv file path')
    argparser.add_argument('-i', '--imgfolder', required=True, type=str, help='images folder')
    argparser.add_argument('-t', '--type', type=int, default=0, help='int: [1..4]')
    argparser.add_argument('-n', '--number', type=int, default=5, help='number of displayed images')
    return argparser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    df = get_reformated_df(args.csv)
    df = extend_df(df, args.imgfolder)
    img_num = len(df)
    defects_num = sum(df['defects'] > 0)

    print('the number of images with no defects: {}'.format(img_num - defects_num))
    print('the number of images with defects: {}'.format(defects_num))
    if 0 < args.type < 5:
        df = df[df[args.type].notna()]
    images_id = df.index
    for image_id in np.random.choice(images_id, args.number, replace=False):
        show_mask(df, args.imgfolder, image_id)
