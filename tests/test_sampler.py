from rtutils.sampler import DeterministicDistributedSampler, InfiniteDistributedSampler
from torch.utils.data import DataLoader
import pytest


@pytest.mark.parametrize("seed", 
    [0, 100]
)
@pytest.mark.parametrize("global_batch_size, batch_size", 
    [[0, 25], [100, 0]]
)
@pytest.mark.parametrize("num_replicas", 
    [4]
)
@pytest.mark.parametrize("rank", 
    [0,1,2,3]
)
@pytest.mark.parametrize("dataset_size", 
    [1000, 1050]
)
def test_deterministic_sampler(seed, global_batch_size, batch_size, num_replicas, rank, dataset_size):
    dataset = list(range(dataset_size))
    x = DeterministicDistributedSampler(dataset, num_replicas, rank, shuffle=True, global_batch_size=global_batch_size, batch_size=batch_size, seed=seed)
    dl = DataLoader(dataset, batch_size=batch_size or global_batch_size // num_replicas, sampler=x, collate_fn=lambda x: x)

    tmp = []
    for epoch in range(10):
        for x in dl:
            tmp.append(x)

    # resume_froms = [5,15,30,44,50]
    resume_froms = [0,10]

    for resume_from in resume_froms:
        x = DeterministicDistributedSampler(dataset, num_replicas, rank, shuffle=True, global_batch_size=global_batch_size, batch_size=batch_size, seed=seed)
        x.set_epoch(resume_from)
        print(resume_from, len(x))
        dl = DataLoader(dataset, batch_size=batch_size or global_batch_size // num_replicas, sampler=x, collate_fn=lambda x: x)
        tmp1 = []
        for epoch in range(10):
            cnt = 0
            for x in dl:
                cnt += 1
                tmp1.append(x)
            if epoch == 0:
                print(cnt)
        for x,y in zip(tmp[resume_from:], tmp1):
            assert x == y


def test_inifinte():
    dataset = list(range(100))
    x = InfiniteDistributedSampler(dataset, 4, 0, shuffle=False)
    dl = DataLoader(dataset, batch_size=22, sampler=x, collate_fn=lambda x: x)
    for x in dl:
        print(x)
        break


@pytest.mark.parametrize("global_batch_size, batch_size", 
    [[0, 4], [16, 0]]
)
def test_inifinte_deterministic(global_batch_size, batch_size):
    dataset = list(range(1000))
    x = InfiniteDistributedSampler(dataset, 4, 0, shuffle=True, deterministic=True, global_batch_size=global_batch_size, batch_size=batch_size)
    dl = DataLoader(dataset, batch_size=4, sampler=x, collate_fn=lambda x: x)

    tmp = []
    for x,_ in zip(dl, range(10)):
        print(x)
        tmp.append(x)

    print('-'*100)
    x = InfiniteDistributedSampler(dataset, 4, 0, shuffle=True, deterministic=True, global_batch_size=global_batch_size, batch_size=batch_size)
    x.set_epoch(2)
    dl = DataLoader(dataset, batch_size=4, sampler=x, collate_fn=lambda x: x)
    for x,_ in zip(dl, range(8)):
        print(x)
        assert x == tmp[_+2]

