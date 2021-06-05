import torch
import itertools
import torch.distributed as dist

class ResumableDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._need_to_resume = False
        self._rng_state = None
        self._iter = None

    # def __iter__(self):
    #     if self._need_to_resume:
    #         torch.set_rng_state(self._rng_state)
    #         tmp_num_workers = self.num_workers
    #         self.num_workers = 0
    #         dataiter = super().__iter__()
    #         import itertools
    #         __sampler_iter, _sampler_iter = itertools.tee(dataiter._sampler_iter)
    #         next(__sampler_iter) # Make sure the _sampler_iter is instantiated

    #         # Run through the saved_iter, make the sampler at the state of halt
    #         for i in range(self._saved_iter):
    #             next(_sampler_iter)

    #         self.num_workers = tmp_num_workers
    #         dataiter = super().__iter__()
    #         dataiter._sampler_iter = _sampler_iter
    #         import time
    #         time.sleep(5)
    #         # drain the prefetched batches
    #         for i in range(dataiter._send_idx - dataiter._rcvd_idx):
    #             next(dataiter)

    #         self._iter = dataiter # Save it in case
    #         self._need_to_resume = False # Only resume at the first iter(loader) after load_state_dict
    #         return self._iter
    #     else:
    #         self._rng_state = torch.get_rng_state()
    #         self._iter = super().__iter__()
    #         return self._iter

    # # save sampler_iter and rng-state
    # def __iter__(self):
    #     if self._need_to_resume:
    #         torch.set_rng_state(self._rng_state)
    #     else:
    #         self._rng_state = torch.get_rng_state()

    #     tmp_num_workers = self.num_workers
    #     self.num_workers = 0
    #     dataiter = super().__iter__()
    #     import itertools
    #     __sampler_iter, _sampler_iter = itertools.tee(dataiter._sampler_iter)
    #     next(__sampler_iter) # Make sure the _sampler_iter is instantiated

    #     self._sampler_iter, _sampler_iter = itertools.tee(_sampler_iter) # Save the full iter

    #     if self._need_to_resume:
    #         # Run through the saved_iter, make the sampler at the state of halt
    #         for i in range(self._saved_iter):
    #             next(_sampler_iter)

    #     if self._need_to_resume:
    #         # Assertion
    #         # assert _sampler_iter the same as saved_sampler:
    #         tmp_sampler_iter, _ = itertools.tee(self._sampler_iter)
    #         for idxs1, idxs2 in zip(tmp_sampler_iter, self._saved_sampler):
    #             assert idxs1 == idxs2

    #     self.num_workers = tmp_num_workers
    #     dataiter = super().__iter__()
    #     dataiter._sampler_iter = _sampler_iter
    #     import time
    #     time.sleep(5)
    #     # drain the prefetched batches
    #     for i in range(dataiter._send_idx - dataiter._rcvd_idx):
    #         next(dataiter)

    #     self._iter = dataiter # Save it in case
    #     self._need_to_resume = False # Only resume at the first iter(loader) after load_state_dict
    #     return self._iter


    def get_sampler_iter(self):
        tmp_num_workers = self.num_workers
        self.num_workers = 0
        dataiter = super().__iter__()
        __sampler_iter, _sampler_iter = itertools.tee(dataiter._sampler_iter)
        next(__sampler_iter) # Make sure the _sampler_iter is instantiated
        self.num_workers = tmp_num_workers
        return _sampler_iter


    def __iter__(self):

        if self._need_to_resume:
            # Run through the saved_iter, make the sampler at the state of halt
            self._sampler_iter, _sampler_iter = itertools.tee(iter(self._saved_sampler))
            for i in range(self._saved_iter):
                next(_sampler_iter)
        else:
            _sampler_iter = self.get_sampler_iter()
            self._sampler_iter, _sampler_iter = itertools.tee(_sampler_iter) # Save the full iter

        self.get_sampler_to_save() # for state_dict


        dataiter = super().__iter__()
        dataiter._sampler_iter = _sampler_iter
        if self.num_workers > 0:
            # Wait for prefetching
            import time
            time.sleep(5)
            # drain the prefetched batches
            for i in range(dataiter._send_idx - dataiter._rcvd_idx):
                next(dataiter)

        self._iter = dataiter # Save it in case
        self._need_to_resume = False # Only resume at the first iter(loader) after load_state_dict
        return self._iter

    def get_sampler_to_save(self):
        # All the indices:
        self._sampler_to_save = []
        for idxs in self._sampler_iter:
            if dist.is_available() and type(self.sampler) is torch.utils.data.distributed.DistributedSampler:
                idxs_list = [torch.zeros(len(idxs)) for k in range(dist.get_world_size())]
                dist.all_gather(idxs_list, torch.tensor(idxs), async_op=False)
                self._sampler_to_save.append([_.tolist() for _ in idxs_list])
            else:
                self._sampler_to_save.append(idxs)


    def state_dict(self):
        if self._iter is None:
            return None
        # The number of batches rest in this epoch
        num_rest = 0
        self._iter._sampler_iter, _sampler_iter = itertools.tee(self._iter._sampler_iter)
        for idxs in _sampler_iter:
            num_rest += 1

        # the number of batched prefetched by the loader
        if self.num_workers > 0:
            num_prefetched = self._iter._send_idx - self._iter._rcvd_idx
        else:
            num_prefetched = 0
        self._saved_iter = len(self) - num_rest - num_prefetched

        return {'saved_iter': self._saved_iter,
                'sampler': self._sampler_to_save}

    # def state_dict(self):
    #     # The number of batches rest in this epoch
    #     num_rest = 0
    #     self._rest_indices = []
    #     for idxs in self._iter._sampler_iter:
    #         num_rest += 1
    #         self._rest_indices.append(idxs)

    #     # the number of batched prefetched by the loader
    #     num_prefetched = self._iter._send_idx - self._iter._rcvd_idx
    #     self._saved_iter = len(self) - num_rest - num_prefetched

    #     return {'saved_iter': self._saved_iter,
    #             'rng_state': self._rng_state,
    #             'rest_indices': self._rest_indices}

    def load_state_dict(self, state_dict):
        if state_dict is None:
            return
        self._need_to_resume = True
        self._saved_iter = state_dict['saved_iter']
        self._saved_sampler = state_dict['sampler']
        if dist.is_available() and type(self.sampler) is torch.utils.data.distributed.DistributedSampler:
            self._saved_sampler = [_[dist.get_rank()] for _ in self._saved_sampler]



if __name__ == '__main__':

    train_loader = ResumableDataLoader(list(range(1000)), #testDataset(),
                                       batch_size=10,
                                       shuffle=True,
                                       num_workers=2)

    
    dataloader_iter = iter(train_loader)

    batches = []
    for i, data in enumerate(dataloader_iter):
        print(i, data)
        batches.append(data)
        if i == 3:
            print('Save the state_dict after the third iteration')
            state_dict = train_loader.state_dict()
        if i == 20:
            print('End up here')
            break

    train_loader.load_state_dict(state_dict)
    print('-----')
    print('Resume from the fourth iteration')
    for i, data in enumerate(train_loader):
        print(i+4, data)
        if i+4 >= len(batches):
            print(i+4)
            break
        assert (batches[i+4] == data).all()

    print('--------------')


    # test 2: edge case, when dataiter has gone to the end. Make sure no error

    train_loader = ResumableDataLoader(list(range(10)), #testDataset(),
                                    batch_size=10,
                                    shuffle=True,
                                    num_workers=2)

    dataloader_iter = iter(train_loader)
    for i, data in enumerate(dataloader_iter):
        print(data)
    state_dict = train_loader.state_dict()
    train_loader.load_state_dict(state_dict)
    for i, data in enumerate(train_loader):
        print(data)

    
    


