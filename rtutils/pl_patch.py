import types
import os
from rtutils.sampler import DeterministicDistributedSampler

import pytorch_lightning as pl
if 'rt' in pl.__author__:
    RTPL = True
else:
    RTPL = False


def assert_RTPL():
    assert RTPL, 'You are not using rt\'s modified pytorch_lightning.\n' \
        'Install pytorch lightning from https://github.com/ruotianluo/pytorch-lightning;\n' \
        'for example, pip install git+https://github.com/ruotianluo/pytorch-lightning or \n' \
        'pip install git+https://github.com/ruotianluo/pytorch-lightning@{tag}'


assert_RTPL()


def _get_distributed_sampler(self, dataloader, shuffle, mode):
    # modified from https://github.com/PyTorchLightning/pytorch-lightning/blob/HEAD/pytorch_lightning/trainer/data_loading.py?q=replace_ddp_sampler#L217
    from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_6
    from pytorch_lightning.overrides.distributed import UnrepeatedDistributedSampler
    from pytorch_lightning.trainer.states import RunningStage
    kwargs = self.distributed_sampler_kwargs
    kwargs["shuffle"] = shuffle and not self.overfit_batches
    if _TORCH_GREATER_EQUAL_1_6:
        kwargs.setdefault("seed", int(os.getenv("PL_GLOBAL_SEED", 0)))
    cls = UnrepeatedDistributedSampler if mode == RunningStage.PREDICTING else DeterministicDistributedSampler
    if cls == DeterministicDistributedSampler:
        kwargs["batch_size"] = dataloader.batch_size
    sampler = cls(dataloader.dataset, **kwargs)

    # rt note here: we set_epoch before, because we want the progress bart start from 0/{remaining batches}, instead of 0/{full batches}. However, we have now a better way to handle progressbar.
    # I just leave a note here to remind myself
    # # If we want to know if it is the end of an epoch, we cannot set_epoch here. because we rely on fixed number of batches each epoch.
    # # sampler.set_epoch(self.total_batch_idx) # we set_epoch here so that the progress bar will know the correct size. In fact pl will set_epoch later and override this. This is just a spoiler for the trainer to know how much batches left for this epoch.
    return sampler


def patch_pl_trainer_with_deterministic_sampler(trainer):
    assert trainer.accelerator_connector.is_distributed, 'Your trainer is not distributed. Cannot replace.'
    assert trainer.accelerator_connector.replace_sampler_ddp, 'Make sure you set replace_sampler_ddp to be True'
    # TODO, we also need to make sure the dataloader is not set some DistributedSampler, otherwise the replace_sampler will not be called either.
    trainer._get_distributed_sampler = types.MethodType(_get_distributed_sampler, trainer)


class ProgressBarPatch(pl.callbacks.ProgressBar):
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        # This manually set the progress bar to the middle according to the total_batch_idx.
        # asssume fix num batches in each epoch.
        self.main_progress_bar.update(trainer.total_batch_idx % trainer.num_training_batches)


class KeyboardInterruptModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_keyboard_interrupt(self, trainer, pl_module):
        # Save model when keyboard interrupt
        filepath = os.path.join(self.dirpath, self.CHECKPOINT_NAME_LAST+'.ckpt')
        if not trainer.total_batch_idx % trainer.num_training_batches == 0:
            # end in the middle of a epoch
            trainer.current_epoch -= 1 # because when saving, it will be increased.
        else:
            # if the model is interrupted at the end of an epoch, for example during validation,
            # the model will be resumed from the beginning of the next epoch.
            pass
        self._save_model(trainer, filepath=filepath)