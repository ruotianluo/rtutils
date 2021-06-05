import os
import pylab

def save_current_fig(fn):
    save_path = os.path.join('/home-nfs/rluo/rluo/figures', fn)
    fig = pylab.gcf()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig.savefig(save_path)
