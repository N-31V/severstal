import os
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_mask(train_df, img_name):
    idxs = train_df.loc[train_df['ImageId'] == img_name].index.to_numpy()
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)
    for idx in idxs:
        mask_label = np.zeros(1600 * 256, dtype=np.uint8)
        label = train_df.loc[idx, 'EncodedPixels'].split(" ")
        positions = map(int, label[0::2])
        length = map(int, label[1::2])
        for pos, le in zip(positions, length):
            mask_label[pos - 1:pos + le - 1] = 1
        mask[:, :, train_df.loc[idx, 'ClassId'] - 1] = mask_label.reshape(256, 1600, order='F')
    return mask


def show_mask(train_df, train_path, img_name):
    palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
    idxs = train_df.loc[train_df['ImageId'] == img_name].index.to_numpy()
    mask = get_mask(train_df, img_name)

    img = cv2.imread(str(train_path + img_name))
    fig, ax = plt.subplots(figsize=(15, 15))

    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    ax.set_title(img_name)
    ax.imshow(img)
    plt.show()


def split_train_val(root_dir, n):
    train_dir = os.path.join('train')
    val_dir = os.path.join('val')
    class_names = os.listdir(root_dir)
    for dir_name in [train_dir, val_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

    for class_name in class_names:
        source_dir = os.path.join(root_dir, class_name)
        for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
            if i % n != 0:
                dest_dir = os.path.join(train_dir, class_name)
            else:
                dest_dir = os.path.join(val_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))


def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
