# RT's utility functions

To install: `pip install rtutils`

Note that: some functions are designed to only work for me and my machines. But those that are useful, I make them as general as possible.
## Stateless resume from checkpoint saved in the middle of one epoch

The change is essencially one line for import and one line for use.

Import function implemented in our repo:
```
import rtutils.pl_patch
```

Use the function imported to modify the pytorch_lightning trainer:
```
trainer = pl.Trainer(...)
# after defining the trainer
rtutils.pl_patch.patch_everything(trainer)
```

And, see the file examples/pl_deterministic_sampler.py to see what other changes need to made. 

Some requirements:
- pytorch_lightning version: >= 1.4.0 for rtutils >= 0.3; pytorch_lightning version: >=1.3 and < 1.4 for rtutils < 0.3
- Using ModelCheckpoint
- Using ddp training
- replace_ddp_sampler is True (default to be True); and you are not manually setting any train loader sampler.
- Fixed batch_size and fixed dataset. (It would also work if batch_size or dataset changes, but it just will not be a "correct" resume.)
- It won't work with your existing checkpoints because the existing checkpoint doesn't have total_batch_idx saved.

Expected behavior:
- If you ctrl-C the training, the trainer will save a "last" checkpoint at current iteration (you can try this with in examples/pl_deterministic_sampler.py now); or you can write your own callback that saves checkpoint at middle of an epoch (e.g. every k iterations).
- Next time when you resume from "last" checkpoint, it will resume from the exact iteration.

To what level of resume:
- Only dataset indices are resumed. It guarantee that you will not see the same data twice in one epoch.
- RNG states will not be resumed. So the dropout or the random augmentation of data in the worker will be different, comparing resume from keep training. (If you don't understand, it's fine. I don't really think it matters.)

## To use drive
visit [link](https://developers.google.com/drive/api/v3/quickstart/python)

Follow this to get the credentials.json.

copy to home directory and rename to  `.rtutils_credentials.json`.
