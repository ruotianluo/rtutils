import torch
from torch.utils.data.distributed import DistributedSampler
import math
import warnings
import itertools


class DeterministicDistributedSampler(DistributedSampler):

    def __init__(self, *args, **kwargs):
        """
        we always start the generator with seed 0,
        and then to restart from certain iteration, we just
        manually drain the iterator. This can make dataloader
        properly resumed.

        Big note: we assume set_epoch takes in the iteration!!!

        Args:
            global_batch_size: We need global_batch_size because we
                we need to know how many interations each epoch so
                that we drain the iterator correctly.
                and we also manually apply drop_last.
            batch_size: local batch size. since we know world size,
                we can drive the global_batch_size
        """
        self.global_batch_size = kwargs.pop('global_batch_size', 0)
        self.batch_size = kwargs.pop('batch_size', 0)

        super().__init__(*args, **kwargs)

        warnings.warn(
            'You are using a customized distributed sampler. Make sure you are feeding '
            'iteration number to the set_epoch function; In addition, batch_size has to be fixed!'
        )

        # intialize batch_size and global_batch_size.
        if self.batch_size != 0:
            assert self.global_batch_size == 0, 'don\'t set both batch_size and global_batch_size'
            self.global_batch_size = self.batch_size * self.num_replicas
        elif self.global_batch_size != 0:
            assert self.batch_size == 0, 'don\'t set both batch_size and global_batch_size'
            self.batch_size = self.global_batch_size // self.num_replicas
        else:
            assert False, 'batch_size or global_batch_size should be specified.'

        # define the generator. There will be one generator form the beginning to the end.
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self._drained = False # if False, we will drain the iterator according to current self.epoch

    def __len__(self):
        if not self._drained:
            iterations_in_one_epoch = self.num_samples // self.batch_size
            done_iteration_in_this_epoch = self.epoch % iterations_in_one_epoch
            return (iterations_in_one_epoch - done_iteration_in_this_epoch) * self.batch_size
        else:
            return self.num_samples // self.batch_size * self.batch_size

    # Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/samplers/distributed_sampler.py#L12
    def __iter__(self):
        start = self.rank
        # drain the sampler
        if not self._drained:
            _i = 0
            _iter = iter(self._indices())
            while _i < self.epoch * self.batch_size:
                try:
                    next(_iter)
                except:
                    _iter = iter(self._indices())
                    next(_iter)
            self._drained = True
            return _iter
        else:
            return iter(self._indices())
    
    def __iter__(self):
        start = self.rank
        indices = self._indices()
        # drain the sampler
        if not self._drained:
            done_epochs = (self.epoch * self.batch_size) // len(indices)
            for i in range(done_epochs):
                indices = self._indices()
            done_iteration_in_this_epoch = (self.epoch * self.batch_size) % len(indices)
            self._drained = True
            return iter(indices[done_iteration_in_this_epoch:])
        else:
            return iter(indices)

    def _indices(self):
        """
        This function almost copy from original pytorch implementation;
        first change: use one generator
        second change: directly apply drop_last here. Because we want to make sure
        """
        if self.shuffle:
            g = self.generator
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # manual drop_last
        indices = indices[:self.num_samples // self.batch_size * self.batch_size]

        return indices


class InfiniteDistributedSampler(DistributedSampler):

    def __init__(self, *args, **kwargs):
        """
        Args:
            global_batch_size: since infinite indices will wrap,
                so it is possible that same images in one batch.
                We apply drop_last here in the sampler.
            determistic: we always start the generator with seed 0,
                and then to restart from certain iteration, we just
                manually drain the iterator. This can make dataloader
                properly resumed.
        """
        self.global_batch_size = kwargs.pop('global_batch_size', 0)
        self.batch_size = kwargs.pop('batch_size', 0)

        self.deterministic = kwargs.pop('deterministic', False)

        super().__init__(*args, **kwargs)

        # intialize batch_size and global_batch_size.
        if self.batch_size != 0:
            assert self.global_batch_size == 0, 'don\'t set both batch_size and global_batch_size'
            self.global_batch_size = self.batch_size * self.num_replicas
        elif self.global_batch_size != 0:
            assert self.batch_size == 0, 'don\'t set both batch_size and global_batch_size'
            self.batch_size = self.global_batch_size // self.num_replicas
        else:
            assert not self.deterministic, 'You have to specify batch_size or global_batch_size if determinstic is True'

    # Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/samplers/distributed_sampler.py#L12
    def __iter__(self):
        start = self.rank
        if self.deterministic:
            for _i, idx in enumerate(itertools.islice(self._infinite_indices(), start, None, self.num_replicas)):
                if _i >= self.epoch * self.batch_size:  # Make sure the epoch is actually iteration
                    yield idx
        else:
            yield from itertools.islice(self._infinite_indices(), start, None, self.num_replicas)

    def _infinite_indices(self):
        g = torch.Generator()
        if self.deterministic:
            g.manual_seed(0)
        else:
            g.manual_seed(self.epoch)
        while True:
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))


            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

            if self.global_batch_size != 0:
                # do what drop_last do, make it devisible by global_batch_size
                indices = indices[:self.total_size // self.global_batch_size * self.global_batch_size]

            yield from indices