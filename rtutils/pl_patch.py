import types
import os
from rtutils.sampler import DeterministicDistributedSampler
import warnings

import pytorch_lightning as pl


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


def patch_progressbar(trainer):
    # This on_train_epoch_start will run after progress_bar on_train_epoch_start
    # this manually set the progress bar to the middle according to the total_batch_idx.
    # asssume fix num batches in each epoch.
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.progress.ProgressBar):
            old_on_train_epoch_start = callback.on_train_epoch_start
            def on_train_epoch_start(self, trainer, pl_module):
                old_on_train_epoch_start(trainer, pl_module)
                self.main_progress_bar.update(trainer.total_batch_idx % trainer.num_training_batches)
            callback.on_train_epoch_start = types.MethodType(on_train_epoch_start, callback)


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


def patch_model_checkpoint(trainer):
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            old_on_keyboard_interrupt = callback.on_keyboard_interrupt
            def on_keyboard_interrupt(self, trainer, pl_module):
                old_on_keyboard_interrupt(trainer, pl_module)
                KeyboardInterruptModelCheckpoint.on_keyboard_interrupt(self, trainer, pl_module)
            callback.on_keyboard_interrupt = types.MethodType(on_keyboard_interrupt, callback)


def patch_checkpoint_connector(trainer):
    # save and load total_batch_idx
    old_restore_training_state = trainer.checkpoint_connector.restore_training_state
    def restore_training_state(self, checkpoint, load_optimizer_states: bool = True):
        old_restore_training_state(checkpoint, load_optimizer_states)
        self.trainer.total_batch_idx = checkpoint.get('total_batch_idx', 0)
    old_dump_checkpoint = trainer.checkpoint_connector.dump_checkpoint
    def dump_checkpoint(self, weights_only: bool = False) -> dict:
        checkpoint = old_dump_checkpoint(weights_only)
        checkpoint['total_batch_idx'] = self.trainer.total_batch_idx
        return checkpoint
    trainer.checkpoint_connector.restore_training_state = types.MethodType(restore_training_state, trainer.checkpoint_connector)
    trainer.checkpoint_connector.dump_checkpoint = types.MethodType(dump_checkpoint, trainer.checkpoint_connector)


class SetEpochCallback(pl.callbacks.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.train_dataloader.sampler.set_epoch(trainer.total_batch_idx)


def patch_set_epoch(trainer):
    old_on_train_epoch_start = trainer.train_loop.on_train_epoch_start
    def on_train_epoch_start(self, epoch):
        old_on_train_epoch_start(epoch)
        self.trainer.train_dataloader.sampler.set_epoch(self.trainer.total_batch_idx)
    trainer.train_loop.on_train_epoch_start = types.MethodType(on_train_epoch_start, trainer.train_loop)


def patch_data_connector(trainer):
    # set enumerate initial batch_idx according to the loader size.
    # copy and rewrite the function
    # def get_profiled_train_dataloader(self, train_dataloader):
    #     # We feed batch_idx because the model may resume of middle-of-epoch checkpoint.
    #     # the length of train_dataloader has been modified by set_epoch at this point.
    #     from pytorch_lightning.trainer.supporters import prefetch_iterator
    #     start_batch_idx = self.trainer.num_training_batches - len(train_dataloader)
    #     profiled_dl = self.trainer.profiler.profile_iterable(
    #         enumerate(prefetch_iterator(train_dataloader), start_batch_idx), "get_train_batch"
    #     )
    #     return profiled_dl
    old_get_profiled_train_dataloader = trainer.data_connector.get_profiled_train_dataloader
    # reuse the old function
    def get_profiled_train_dataloader(self, train_dataloader):
        # We feed batch_idx because the model may resume of middle-of-epoch checkpoint.
        # the length of train_dataloader has been modified by set_epoch at this point.
        start_batch_idx = self.trainer.num_training_batches - len(train_dataloader)
        old_profiled_dl = old_get_profiled_train_dataloader(train_dataloader)
        def profiled_dl():
            # we discard the old batch_idx, and use the new one.
            for batch_idx, (_, batch) in enumerate(old_profiled_dl, start_batch_idx):
                yield batch_idx, batch
        return profiled_dl()
    trainer.data_connector.get_profiled_train_dataloader = types.MethodType(get_profiled_train_dataloader, trainer.data_connector)


def patch_everything(trainer):
    warnings.warn('Patch everything will replace trainer\'s (and its members\') functions with something else. In general it should be fine. But make sure you are confortable with this.')
    patch_model_checkpoint(trainer)
    patch_set_epoch(trainer)
    patch_progressbar(trainer)
    patch_pl_trainer_with_deterministic_sampler(trainer)
    patch_checkpoint_connector(trainer)
    patch_data_connector(trainer)


def patch_everything_safer(trainer):
    """
    Compare to patch_everything, we remove the patches that can be implemented by callbacks.
    This need to work with all_callbacks() and KeyboardInterruptModelCheckpoint to work properly.

    For examples:
    callbacks = [
        KeyboardInterruptModelCheckpoint(...),
        *rtutils.pl_patch.all_callbacks(),
    ]
    trainer = pl.Trainer(
        ...,
        callbacks = callbacks,
        ...
    )
    rtutils.pl_patch.patch_everything_safer(trainer)
    """
    warnings.warn('Patch everything will replace trainer\'s (and its members\') functions with something else. In general it should be fine. But make sure you are confortable with this.')
    warnings.warn('To fully function, you would also want to include all the callbacks in pl_path.all_backs and use KeyboardInterruptModelCheckpoint to create ModelCheckpoint Instance')
    patch_pl_trainer_with_deterministic_sampler(trainer)
    patch_checkpoint_connector(trainer)
    patch_data_connector(trainer)


def all_callbacks():
    return [
        ProgressBarPatch(),
        SetEpochCallback(),
    ]