import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms


class SteelDataset(Dataset):
    def __init__(self, data_folder, df=None):
        if df is None:
            idx = os.listdir(data_folder)
            df = pd.DataFrame(index=idx, columns=[1, 2, 3, 4])
        self.df = df
        self.root = data_folder
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.names = self.df.index.tolist()

    def __getitem__(self, row_id):
        image_id = self.df.index[row_id]
        mask = get_mask(self.df, image_id)
        image_path = os.path.join(self.root, image_id)
        img = cv2.imread(image_path)
        img = self.transforms(img)
        mask = torch.from_numpy(mask)  # 256x1600x4
        mask = mask.permute(2, 0, 1)  # 4x256x1600
        return img, mask, image_id

    def __len__(self):
        return len(self.names)


def train_val_dataloader(data_folder, df_path, batch_size, num_workers):
    df = extend_df(get_reformated_df(df_path), data_folder)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['defects'])
    train_dataloader = DataLoader(
        SteelDataset(data_folder, train_df),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        SteelDataset(data_folder, val_df),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return {'train': train_dataloader, 'val': val_dataloader}


def get_reformated_df(df_path):
    df = pd.read_csv(df_path)
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    return df


def extend_df(df, data_folder):
    all_idx = os.listdir(data_folder)
    current_idx = df.index.tolist()
    new_idx = list(set(all_idx)-set(current_idx))
    new_df = pd.DataFrame(index=new_idx, columns=[1, 2, 3, 4])
    new_df['defects'] = new_df.count(axis=1)
    df = pd.concat([df, new_df], axis=0)
    return df


def get_mask(df, image_id, dtype=np.float32):
    labels = df.loc[image_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=dtype)
    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(' ')
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=dtype)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return masks


def show_mask(df, path, image_id):
    palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
    masks = get_mask(df, image_id, dtype=np.uint8)
    img = cv2.imread(os.path.join(path, image_id))
    fig, ax = plt.subplots(figsize=(15, 15))
    title = image_id + ' '
    for ch in range(4):
        contours, _ = cv2.findContours(masks[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
            title += str(ch + 1)
    ax.set_title(title)
    ax.imshow(img)
    plt.show()


def pmask_to_binary(out, threshold=0.5):
    """out is sigmoid output of the model"""
    out_b = np.copy(out)
    masks = (out_b > threshold).astype('uint8')
    return masks


def mask_to_output(img):
    """
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def output_to_df(imges_id, predictions):
    concat_list = []
    for img, pred in zip(imges_id, predictions):
        pred = pmask_to_binary(pred)
        for i in range(4):
            if pred[i].sum() > 0:
                concat_list.append(pd.Series(data=[img, mask_to_output(pred[i]), i], index=['ImageId', 'EncodedPixels', 'ClassId']))
    df = pd.concat(concat_list, axis=1).T
    df.set_index('ImageId', inplace=True)
    return df


