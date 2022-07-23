import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from pathlib import Path
from typing import Tuple
import pytorch_lightning as pl

sys.path.append('/data1/hulh/workplace/Kaggle-monai')
sys.path.append('/data1/hulh/workplace/Kaggle-monai/module')

from module.LitModule import LitModule
from module.LitDataModule import LitDataModule

KAGGLE_DIR = Path("/data1/hulh/dataset/Kaggle/HuBMAP")

# INPUT_DIR = KAGGLE_DIR / "input"
# OUTPUT_DIR = KAGGLE_DIR / "working"

COMPETITION_DATA_DIR = KAGGLE_DIR

TRAIN_PREPARED_CSV_PATH = "/data1/hulh/dataset/Kaggle/HuBMAP/train_prepared.csv"
VAL_PRED_PREPARED_CSV_PATH = "/data1/hulh/dataset/Kaggle/HuBMAP/val_pred_prepared.csv"
TEST_PREPARED_CSV_PATH = "/data1/hulh/dataset/Kaggle/HuBMAP/test_prepared.csv"

N_SPLITS = 4
RANDOM_SEED = 2022
SPATIAL_SIZE = 1024
VAL_FOLD = 3
NUM_WORKERS = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
FAST_DEV_RUN = False
GPUS = 1
MAX_EPOCHS = 10
PRECISION = 16
DEBUG = False

DEVICE = "cuda:6"
THRESHOLD = 0.5


def train(
        random_seed: int = RANDOM_SEED,
        train_csv_path: str = str(TRAIN_PREPARED_CSV_PATH),
        test_csv_path: str = str(TEST_PREPARED_CSV_PATH),
        spatial_size: Tuple[int, int] = SPATIAL_SIZE,
        val_fold: str = VAL_FOLD,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        fast_dev_run: bool = FAST_DEV_RUN,
        gpus: int = GPUS,
        max_epochs: int = MAX_EPOCHS,
        precision: int = PRECISION,
        debug: bool = DEBUG,
) -> None:
    pl.seed_everything(random_seed)

    data_module = LitDataModule(
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        spatial_size=spatial_size,
        val_fold=val_fold,
        batch_size=2 if debug else batch_size,
        num_workers=num_workers,
    )

    module = LitModule(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        limit_train_batches=0.1 if debug else 1.0,
        limit_val_batches=0.1 if debug else 1.0,
        log_every_n_steps=5,
        logger=pl.loggers.CSVLogger(save_dir='../output/logs/'),
        max_epochs=2 if debug else max_epochs,
        precision=precision,
    )

    trainer.fit(module, datamodule=data_module)

    return trainer


if __name__ == '__main__':
    # nrows = 3
    #
    # data_module = LitDataModule(
    #     train_csv_path=TRAIN_PREPARED_CSV_PATH,
    #     test_csv_path=TEST_PREPARED_CSV_PATH,
    #     spatial_size=SPATIAL_SIZE,
    #     val_fold=VAL_FOLD,
    #     batch_size=nrows ** 2,
    #     num_workers=NUM_WORKERS,
    # )
    # data_module.setup()

    trainer = train(
        batch_size=32
    )
