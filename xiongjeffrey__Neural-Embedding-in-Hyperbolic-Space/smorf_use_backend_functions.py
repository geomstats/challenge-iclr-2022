# %%
import sys
sys.path.insert(0,'../')

import numpy as np
import pickle
import pandas as pd
from scipy.stats import mode

from edit_distance.task.dataset_generator_genomic import EditDistanceGenomicDatasetGenerator
from hypersmorf.myfunctions import create_parser, generate_datasets,run_model

# %%
# write subset for all and largest group into .txt file
df = pd.read_csv("./datasets/dataset_FINAL.tsv", sep='\t')

subset_groups={}


subset_groups['clean_strings']=df['smorf'].str.contains(r'^[ACTG]+$', na=False)

smorfams=df.clust[df.clust.str.startswith('smorfam') & df.y.str.fullmatch('positive')]
md,count=mode(smorfams)
subset_groups['largest_group_strings']=df.clust.str.startswith(md[0]) & df.y.str.fullmatch('positive') & subset_groups['clean_strings']
'''
for name, boo_list in subset_groups.items():
    with open('./datasets/'+name+'.txt', 'w') as f_out:
        f_out.writelines("%s\n" % l for l in df[boo_list].smorf.values)

# %%
# generate train-test-val splits for select datasets and pickle them
subsets={"strings_test":100,"strings_subset":7000,"clean_strings":35000}

for n, train_size in subsets.items():
    parser=create_parser('./datasets/'+n+'.pkl','./datasets/clean_strings.txt',  train_size,round(train_size/10),round(train_size/5))
    generate_datasets(parser)

train_size=50
parser=create_parser('./datasets/largest_group_strings.pkl','./datasets/largest_group_strings.txt',  train_size,round(train_size/10),round(train_size/5))
generate_datasets(parser)

'''
#%%
#  Train my models

string_size=153
n_epoch=20
e_size=np.logspace(1,9,num=9-1, base=2,endpoint=False, dtype=int)

dist_types=['hyperbolic', 'euclidean']

model, avg_loss=np.zeros((len(dist_types),len(e_size)),dtype=object),np.zeros((len(dist_types),len(e_size)))

names=['largest_group_strings', 'strings_test', 'strings_subset','clean_strings']


for name in names:
    dataset_name='./datasets/'+name+'.pkl'
    for i in range(len(dist_types)):
        for j in range(len(e_size)):
            model[i][j],avg_loss[i][j]=run_model(dataset_name,e_size[j],dist_types[i],string_size,n_epoch)
    pickle.dump((model,avg_loss,e_size,dist_types), open('./models/'+name+'.pkl', "wb"))


# %%
