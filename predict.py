#!/usr/bin/python3

import argparse
from work_with_data import output_to_df
from work_with_model import ModelToolkit, load_model


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--model', type=str, required=True, help='model path')
    return argparser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    model = load_model(args.model)
    img_id, pred = model.predict()
    pred_df = output_to_df(img_id, pred)
    pred_df.to_csv('test.csv')

