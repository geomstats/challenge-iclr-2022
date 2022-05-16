# %% [markdown]
# ### Takes my smorf dataset and uses NeuroSEED's code to store it in the same way

# %%
# Import code

import sys
sys.path.insert(0,'../')

from scipy.stats import mode
import argparse
import pickle
import pandas as pd

from edit_distance.task.dataset_generator_genomic import EditDistanceGenomicDatasetGenerator

# %% [markdown]
# 

# %%
# save dataset as string
df = pd.read_csv("../datasets/dataset_FINAL.tsv", sep='\t')

file_out='../datasets/strings.txt'

no_weird_chars=df['smorf'].str.contains(r'^[ACTG]+$', na=False)

with open(file_out, 'w') as f_out:
    f_out.writelines("%s\n" % l for l in df[no_weird_chars].smorf.values)



# %%
# create string to be read by NeuroSEED-borrowed code
file_out='../datasets/strings.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, default="../datasets/strings.pkl", help='Output data path')
parser.add_argument('--train_size', type=int, default=35000, help='Training sequences')
parser.add_argument('--val_size', type=int, default=3500, help='Validation sequences')
parser.add_argument('--test_size', type=int, default=7000, help='Test sequences')
parser.add_argument('--source_sequences', type=str, default=file_out, help='Sequences data path')
args, unknown = parser.parse_known_args()

# load and divide sequences
with open(args.source_sequences, 'rb') as f:
    L = f.readlines()
L = [l[:-1].decode('UTF-8') for l in L]

strings = {
    'train': L[:args.train_size],
    'val': L[args.train_size:args.train_size + args.val_size],
    'test': L[args.train_size + args.val_size:args.train_size + args.val_size + args.test_size]
}

data = EditDistanceGenomicDatasetGenerator(strings=strings)
data.save_as_pickle(args.out)

