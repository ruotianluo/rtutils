# RT's utility functions

To install: `pip install rtutils`

Note that: some functions are designed to only work for me and my machines. But those that are useful, I make them as general as possible.
## Stateless resume from checkpoint saved in the middle of one epoch

The change is essencially one line.

```
trainer = pl.Trainer(...)
# after defining the trainer
trainer.pl_patch.patch_everything(trainer)
```

And, see the file examples/pl_deterministic_sampler.py to see what other changes need to made. 

## To use drive
visit [link](https://developers.google.com/drive/api/v3/quickstart/python)

Follow this to get the credentials.json.

copy to home directory and rename to  `.rtutils_credentials.json`.
