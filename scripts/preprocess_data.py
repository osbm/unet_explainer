import pandas as pd
import SimpleITK as sitk
import matplotlib.image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import os

dspath = Path("data-raw/")

train_df = pd.read_csv("data-raw/train.csv")
valid_df = pd.read_csv("data-raw/valid.csv")
test_df = pd.read_csv("data-raw/test.csv")

train_df["image"] = train_df["t2"]
valid_df["image"] = valid_df["t2"]
test_df["image"] = test_df["t2"]
train_df["mask"] = train_df["t2_anatomy_reader1"]
valid_df["mask"] = valid_df["t2_anatomy_reader1"]
test_df["mask"] = test_df["t2_anatomy_reader1"]



def normalize(x):
    """Normalize the volume"""
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm


def process_df(df, output_dir):

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "image", exist_ok=True)
    os.makedirs(output_dir / "mask", exist_ok=True)
    imageid=0
    for patient_index, row in tqdm(df.iterrows(), total=len(df), disable=True):
        mask_path = dspath / row["mask"]
        image_path = dspath / row["image"]
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)

        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)

        image = normalize(image)

        mask_idx = 0
        for i in range(mask.shape[0]):
            mask_array = mask[i]
            image_array = image[i]

            # print labels to background ratio
            existing_labels, counts = np.unique(mask_array, return_counts=True)

            if len(existing_labels) != 3:
                continue
            
            label_ratio = counts[1:].sum() / (mask_array.shape[0] * mask_array.shape[1])

            if label_ratio < 0.08:
                continue

            print(imageid)
            imageid+=1
            # matplotlib.image.imsave(output_dir / "image" / f'patient_{patient_index}_{mask_idx}.png', image_array, cmap="gray")
            # matplotlib.image.imsave(output_dir / "mask" / f'patient_{patient_index}_{mask_idx}.png', mask_array, cmap="gray")
            mask_idx += 1

process_df(train_df, "data/train")
process_df(valid_df, "data/valid")
process_df(test_df, "data/test")
