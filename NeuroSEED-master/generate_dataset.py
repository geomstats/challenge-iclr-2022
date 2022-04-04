# %% [markdown]

import sys
sys.path.insert(0,'../')
from scipy.stats import mode
import pandas as pd
from edit_distance.task.dataset_generator_genomic import EditDistanceGenomicDatasetGenerator
from hypersmorf.myfunctions import create_parser, generate_datasets

# %%
# write subset for all and largest group iinto .txt file
df = pd.read_csv("./datasets/dataset_FINAL.tsv", sep='\t')

subset_groups={}


subset_groups['clean_strings']=df['smorf'].str.contains(r'^[ACTG]+$', na=False)

smorfams=df.clust[df.clust.str.startswith('smorfam') & df.y.str.fullmatch('positive')]
md,count=mode(smorfams)
subset_groups['largest_group_strings']=df.clust.str.startswith(md[0]) & df.y.str.fullmatch('positive') & subset_groups['clean_strings']

for name, boo_list in subset_groups.items():
    with open('./datasets/'+name+'.txt', 'w') as f_out:
        f_out.writelines("%s\n" % l for l in df[boo_list].smorf.values)

# %%
# generate train-test-val splits for select datasets and pickle them
subsets={"strings_test":100,"strings_subset":7000,"strings":35000}

for n, train_size in subsets.items():
    parser=create_parser('./datasets/'+n+'.pkl','./datasets/clean_strings.txt',  train_size,round(train_size/10),round(train_size/5))
    generate_datasets(parser)

# %%
