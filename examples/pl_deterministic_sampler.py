# Example of pl_deterministic_sampler.

import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer

# Note:
# change are unnecessary changes, just for demoonstration in this file
# CHANGE are necessar changes.


# This file is modified from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/bug_report_model.py

################################################################
# To use my deterministic sampler and pytorch lightning integration,
# there is a few things you need to know.
# First, you have to use my fork of the pytorch-lightning which includes some changes.
# there are no breaking changes and should work as the original version in other cases.
# Second, this sampler assumes fixed batch_size and fixed number of iterations in one epoch.
################################################################

import rtutils.pl_patch  # CHANGE

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        # change
        # sleep to make training slower for you have time to interrupt, do not follow this in practice.
        import time; time.sleep(1)
        # change end
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        # change
        # sleep to make training slower for you have time to interrupt, do not follow this in practice.
        import time; time.sleep(0.5)
        # change end
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    # CHANGE
    callbacks = [
        rtutils.pl_patch.KeyboardInterruptModelCheckpoint(
            dirpath=os.getcwd(),
            save_last=True,
        ),  # the model checkpoint will save the checkpoint when KeyInterrupt is raised.
        rtutils.pl_patch.ProgressBarPatch(),  # the progressbar will start from where you stopped
    ]
    # CHANGE end

    model = BoringModel()
    trainer = Trainer(
        callbacks = callbacks,
        default_root_dir=os.getcwd(),
        accelerator='ddp',  # CHANGE; the deterministic sampler only work for ddp training.
        gpus=1, # change
        resume_from_checkpoint='last.ckpt' if os.path.exists('last.ckpt') else None,  # CHANGE; resume from middle of checkpoint
        # limit_train_batches=1,  # change
        # limit_val_batches=1,  # change
        num_sanity_val_steps=0,
        max_epochs=2,
        weights_summary=None,
    )
    rtutils.pl_patch.patch_pl_trainer_with_deterministic_sampler(trainer)  # CHANGE; when replacing sampler, the training sampler will be replaced by my deterministic sampler.
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)
    # trainer.test(model, test_dataloaders=test_data) # change


if __name__ == '__main__':
    run()