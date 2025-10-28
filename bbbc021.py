from typing import Callable

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import v2

# You can get bbbc021_df_all.csv and the preprocessed images
# from the IMPA reproducibility drop: https://zenodo.org/records/8307629
# We assume you have added the following additional columns to the CSV:
# - path: path to where you saved each image as a .npy file
# - experiment_id: unique ID for each experiment (biological batch)
# - perturbation_id: unique ID for each perturbation
# - perturbation_id_with_dose: unique ID for each perturbation + dose combination
METADATA_PATH: str = ...  # Fill this in with the path to the CSV

IMG_SIZE = 96
IMG_CHANNELS = 3

# no harm in these being larger than the actual number of unique labels
# since it will just be used to create one-hot label embeddings
NUM_PERTURBATIONS = 128
NUM_EXPERIMENTS = 64
NUM_CELL_TYPES = 1


def uniform_dq(img, scale=1 / 128.0):
    """Dequantize by adding uniform noise to the input.
    This is a common trick to improve performance of generative models.
    Explained well in Sec. 3.1.1. of arxiv.org/abs/1902.00275."""
    return img + (torch.rand_like(img) * scale)


def encode(x):
    """torch.uint8 [0,255) -> torch.float32 [-1,1]"""
    return x.to(torch.float32) / 127.5 - 1


def decode(x):
    """torch.float32 [-1,1] -> torch.uint8 [0,255)"""
    return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


def get_transforms() -> Callable[[torch.Tensor], torch.Tensor]:
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ]
    )


class CellDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata: pl.DataFrame,
        train: bool,
    ):
        required_cols = [
            "path",
            "experiment_id",
            "perturbation_id",
            "perturbation_id_with_dose",
        ]
        assert all(col in metadata.columns for col in required_cols), (
            f"metadata must have the following columns: {required_cols}"
        )
        self.metadata = metadata
        self.transforms = get_transforms()
        self.train = train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.row(index, named=True)
        pert_path = row["path"]
        perturbation_id = row["perturbation_id_with_dose"]
        experiment_id = row["experiment_id"]

        pert_array = np.load(pert_path)  # np.uint8 (96x96x3)
        pert_tensor = torch.from_numpy(pert_array).permute(2, 0, 1)  # uint8 (3x96x96)

        with torch.inference_mode():
            pert_tensor = encode(pert_tensor)  # float32 [-1,1]

            if self.train:
                pert_tensor = self.transforms(pert_tensor)
                pert_tensor = uniform_dq(pert_tensor)

        return {
            "pert": pert_tensor,
            "perturbation_id": perturbation_id,
            "experiment_id": experiment_id,
            "cell_type_id": 0,  # all same cell type
        }


def get_dataloaders(
    *,
    path: str = METADATA_PATH,
    batch_size: int = 16,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    df = pl.read_csv(path)
    train_df = df.filter(pl.col("SPLIT") == "train")
    iid_df = df.filter(pl.col("SPLIT") == "iid")
    ood_df = df.filter(pl.col("SPLIT") == "ood")

    train_ds = CellDataset(train_df, train=True)
    iid_ds = CellDataset(iid_df, train=False)
    ood_ds = CellDataset(ood_df, train=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    iid_sampler = DistributedSampler(iid_ds, shuffle=False)
    ood_sampler = DistributedSampler(ood_ds, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
        drop_last=True,
    )
    iid_loader = DataLoader(
        iid_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=iid_sampler,
        pin_memory=pin_memory,
        drop_last=False,
    )
    ood_loader = DataLoader(
        ood_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=ood_sampler,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, iid_loader, ood_loader
