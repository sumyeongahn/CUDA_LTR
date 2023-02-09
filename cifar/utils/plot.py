import matplotlib.pyplot as plt
import seaborn as sns
import torch as t
import numpy as np
import os
import pandas as pd

sns.set_palette("bright")
sns.set_style("darkgrid")


def plot_score_epoch(curr_state, label, epoch, maps, out, name='heat'):
    label = t.tensor(label)
    
    num_samples_per_class = t.sum(t.nn.functional.one_hot(label, num_classes = len(t.unique(label))), dim=0)
    num_samples_sort = t.argsort(num_samples_per_class)
    
    for cidx in t.unique(label):
        pos = t.where(cidx == label)
        maps[epoch,cidx] = t.mean(curr_state[pos]).numpy()

    # sns.heatmap(maps,cmap='YlGnBu', vmin=0, vmax=10)
    # plt.xlabel('Class index')
    # plt.ylabel('Epoch')

    # os.makedirs(f'{out}/score_epoch_plot/', exist_ok=True)
    # plt.savefig(f'{out}/score_epoch_plot/{name}.png')
    # plt.close()
    
    return maps
            
    

