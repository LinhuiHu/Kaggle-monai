from typing import Dict

import monai
import pytorch_lightning as pl

import torch
import torch.nn as nn


class LitModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = self._init_model()

        self.loss_fn = self._init_loss_fn()

        # TODO: add metric

    def _init_model(self) -> nn.Module:
        # TODO: try other networks
        return monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def _init_loss_fn(self):
        # TODO: try other losses
        return monai.losses.DiceLoss(sigmoid=True)

    def configure_optimizers(self):
        # TODO: try other optimizers and schedulers
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        images, masks = batch["image"], batch["mask"]
        outputs = self(images)

        loss = self.loss_fn(outputs, masks)

        self.log("train_loss", loss, batch_size=images.shape[0])

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        images, masks = batch["image"], batch["mask"]
        outputs = self(images)

        loss = self.loss_fn(outputs, masks)

        self.log("val_loss", loss, prog_bar=True, batch_size=images.shape[0])

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path: str, device: str) -> nn.Module:
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module