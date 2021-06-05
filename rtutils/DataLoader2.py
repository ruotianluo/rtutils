import torch
import itertools
import time
import torch.distributed as dist

class ResumableDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._need_to_resume = False
        self._rng_state = None
        self._iter = None

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
                idxs_list = [torch.zeros(len(idxs)).long().to(torch.cuda.current_device()) for k in range(dist.get_world_size())]
                dist.all_gather(idxs_list, torch.tensor(idxs).long().to(torch.cuda.current_device()), async_op=False) # can also use detectron.utils.comm.all_gather
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

    def load_state_dict(self, state_dict):
        if state_dict is None:
            return
        self._need_to_resume = True
        self._saved_iter = state_dict['saved_iter']
        self._saved_sampler = state_dict['sampler']
        if dist.is_available() and type(self.sampler) is torch.utils.data.distributed.DistributedSampler:
            self._saved_sampler = [_[dist.get_rank()] for _ in self._saved_sampler]



def test1():
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

def test2():

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

def main_worker(rank, ngpus, port):

    dist.init_process_group(
            world_size=ngpus, rank=rank,
            backend='nccl', init_method='tcp://127.0.0.1:%d' %port,
    )
    torch.cuda.set_device(rank)

    train_dataset = list(range(10000))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = ResumableDataLoader(
        train_dataset, batch_size=10, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler)

    dataloader_iter = iter(train_loader)

    batches = []
    for i, data in enumerate(dataloader_iter):
        print(rank, i, data)
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
        print(rank, i+4, data)
        if i+4 >= len(batches):
            print(i+4)
            break
        assert (batches[i+4] == data).all()

def test3():
    # distributed:
    import torch.distributed as dist
    import torch.utils.data.distributed
    import torch.multiprocessing as mp
    
    port = 11000
    mp.spawn(main_worker, nprocs=2, args=(2, port))


if __name__ == '__main__':
    test3()





