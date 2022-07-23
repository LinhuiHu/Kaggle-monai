from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import monai
import numpy as np
import pandas as pd
# import pytorch_lightning as pl
# import seaborn as sns
import tifffile
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from monai.data import CSVDataset
from monai.data import DataLoader
from monai.data import ImageReader
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm

KAGGLE_DIR = Path("/data1/hulh/dataset/Kaggle/HuBMAP")

# INPUT_DIR = KAGGLE_DIR / "input"
# OUTPUT_DIR = KAGGLE_DIR / "working"
#
COMPETITION_DATA_DIR = KAGGLE_DIR
#
# TRAIN_PREPARED_CSV_PATH = "train_prepared.csv"
# VAL_PRED_PREPARED_CSV_PATH = "val_pred_prepared.csv"
# TEST_PREPARED_CSV_PATH = "test_prepared.csv"

N_SPLITS = 4
RANDOM_SEED = 2022
SPATIAL_SIZE = 1024
VAL_FOLD = 0
NUM_WORKERS = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
FAST_DEV_RUN = False
GPUS = 1
MAX_EPOCHS = 10
PRECISION = 16
DEBUG = False

DEVICE = "cuda"
THRESHOLD = 0.5

def add_path_to_df(df: pd.DataFrame, data_dir: Path, type_: str, stage: str) -> pd.DataFrame:
    ending = ".tiff" if type_ == "image" else ".npy"

    dir_ = str(data_dir / f"{stage}_{type_}s")
    df[type_] = dir_ + "/" + df["id"].astype(str) + ending
    return df


def add_paths_to_df(df: pd.DataFrame, data_dir: Path, stage: str) -> pd.DataFrame:
    df = add_path_to_df(df, data_dir, "image", stage)
    df = add_path_to_df(df, data_dir, "mask", stage)
    return df


def create_folds(df: pd.DataFrame, n_splits: int, random_seed: int) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["organ"])):
        df.loc[val_idx, "fold"] = fold

    return df


def prepare_data(data_dir: Path, stage: str, n_splits: int, random_seed: int) -> None:
    df = pd.read_csv(data_dir / f"{stage}.csv")
    df = add_paths_to_df(df, data_dir, stage)

    if stage == "train":
        df = create_folds(df, n_splits, random_seed)

    filename = f"{data_dir}/{stage}_prepared.csv"
    df.to_csv(filename, index=False)

    print(f"Created {filename} with shape {df.shape}")

    return df

def rle2mask(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    Source: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def save_array(file_path: str, array: np.ndarray) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)


def save_masks(df: pd.DataFrame) -> None:
    for row in df.itertuples():
        mask = rle2mask(row.rle, shape=(row.img_width, row.img_height))
        save_array(row.mask, mask)

if __name__ == '__main__':
    train_df = prepare_data(COMPETITION_DATA_DIR, "train", N_SPLITS, RANDOM_SEED)
    test_df = prepare_data(COMPETITION_DATA_DIR, "test", N_SPLITS, RANDOM_SEED)
    save_masks(train_df)
