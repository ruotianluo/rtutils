from rtutils.sampler import DeterministicDistributedSampler
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