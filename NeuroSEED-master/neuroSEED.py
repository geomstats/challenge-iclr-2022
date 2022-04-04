import torch
import os 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import sys
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Levenshtein import distance as levenshtein_distance

from edit_distance.train import load_edit_distance_dataset
from util.data_handling.data_loader import get_dataloaders
from util.ml_and_math.loss_functions import AverageMeter

import numpy as np
import pickle
import pandas as pd
from scipy.stats import mode

from edit_distance.task.dataset_generator_genomic import EditDistanceGenomicDatasetGenerator
# from hypersmorf.myfunctions import create_parser, generate_datasets, run_model

import torch
import torch.nn as nn
import numpy as np
from geomstats.geometry.poincare_ball import PoincareBall

numpy_type_map = {
     'float64': torch.DoubleTensor,
     'float32': torch.FloatTensor,
     'float16': torch.HalfTensor,
     'int64': torch.LongTensor,
     'int32': torch.IntTensor,
     'int16': torch.ShortTensor,
     'int8': torch.CharTensor,
     'uint8': torch.ByteTensor,
 }

def square_distance(t1_emb, t2_emb,scale=1):
    D = t1_emb - t2_emb
    d = torch.sum(D * D, dim=-1)
    return d


def euclidean_distance(t1_emb, t2_emb,scale=1):
    D = t1_emb - t2_emb
    d = torch.norm(D, dim=-1)
    return d


def cosine_distance(t1_emb, t2_emb,scale=1):
    return 1 - nn.functional.cosine_similarity(t1_emb, t2_emb, dim=-1, eps=1e-6)


def manhattan_distance(t1_emb, t2_emb,scale=1):
    D = t1_emb - t2_emb
    d = torch.sum(torch.abs(D), dim=-1)
    return d

def hyperbolic_geomstats_distance(u,v,scale=1):
    return PoincareBall(u.size()[1]).metric.dist(u,v)
    

def hyperbolic_distance(u, v, epsilon=1e-7):  # changed from epsilon=1e-7 to reduce error
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = torch.sqrt(x ** 2 - 1)
    return torch.log(x + z)


def hyperbolic_distance_numpy(u, v, epsilon=1e-9):
    sqdist = np.sum((u - v) ** 2, axis=-1)
    squnorm = np.sum(u ** 2, axis=-1)
    sqvnorm = np.sum(v ** 2, axis=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = np.sqrt(x ** 2 - 1)
    return np.log(x + z)


DISTANCE_TORCH = {
    'square': square_distance,
    'euclidean': euclidean_distance,
    'cosine': cosine_distance,
    'manhattan': manhattan_distance,
    'hyperbolic': hyperbolic_distance
}

import argparse
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from edit_distance.task.dataset import EditDistanceDatasetSampled, EditDistanceDatasetComplete,EditDistanceDatasetSampledCalculated
from edit_distance.task.dataset import EditDistanceDatasetCompleteCalculated
from edit_distance.models.hyperbolics import RAdam
from edit_distance.models.pair_encoder import PairEmbeddingDistance
from util.data_handling.data_loader import get_dataloaders
from util.ml_and_math.loss_functions import MAPE
from util.ml_and_math.loss_functions import AverageMeter


def general_arg_parser():
    """ Parsing of parameters common to all the different models """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../../data/edit_qiita_small.pkl', help='Dataset path')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training (GPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
    parser.add_argument('--patience', type=int, default=50, help='Patience')
    parser.add_argument('--print_every', type=int, default=1, help='Print training results every')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=5, help='Size of embedding')
    parser.add_argument('--distance', type=str, default='hyperbolic', help='Type of distance to use')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--loss', type=str, default="mse", help='Loss function to use (mse, mape or mae)')
    parser.add_argument('--plot', action='store_true', default=False, help='Plot real vs predicted distances')
    parser.add_argument('--closest_data_path', type=str, default='', help='Dataset for closest string retrieval tests')
    parser.add_argument('--hierarchical_data_path', type=str, default='', help='Dataset for hierarchical clustering')
    parser.add_argument('--construct_msa_tree', type=str, default='False', help='Whether to construct NJ tree testset')
    parser.add_argument('--extr_data_path', type=str, default='', help='Dataset for further edit distance tests')
    parser.add_argument('--scaling', type=str, default='False', help='Project to hypersphere (for hyperbolic)')
    parser.add_argument('--hyp_optimizer', type=str, default='Adam', help='Optimizer for hyperbolic (Adam or RAdam)')
    return parser


def execute_train(model_class, model_args, args):
    # set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cpu'
    print('Using device:', device)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load data
    datasets = load_edit_distance_dataset(args.data)
    loaders = get_dataloaders(datasets, batch_size=args.batch_size, workers=args.workers)

    # fix hyperparameters
    model_args = SimpleNamespace(**model_args)
    model_args.device = device
    model_args.len_sequence = datasets['train'].len_sequence
    model_args.embedding_size = args.embedding_size
    model_args.dropout = args.dropout
    print("Length of sequence", datasets['train'].len_sequence)
    args.scaling = True if args.scaling == 'True' else False

    # generate model
    embedding_model = model_class(**vars(model_args))
    model = PairEmbeddingDistance(embedding_model=embedding_model, distance=args.distance, scaling=args.scaling)
    model.to(device)

    # select optimizer
    if args.distance == 'hyperbolic' and args.hyp_optimizer == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # select loss
    loss = None
    if args.loss == "mse":
        loss = nn.MSELoss()
    elif args.loss == "mae":
        loss = nn.L1Loss()
    elif args.loss == "mape":
        loss = MAPE

    # print total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params', total_params)

    # Train model
    t_total = time.time()
    bad_counter = 0
    best = 1e10
    best_epoch = -1
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        t = time.time()
        loss_train = train(model, loaders['train'], optimizer, loss, device)
        loss_val = test(model, loaders['val'], loss, device)

        # print progress
        if epoch % args.print_every == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.6f}'.format(loss_train),
                  'loss_val: {:.6f} MAPE {:.4f}'.format(*loss_val),
                  'time: {:.4f}s'.format(time.time() - t))
            sys.stdout.flush()

        if loss_val[0] < best:
            # save current model
            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            # remove previous model
            if best_epoch >= 0:
                os.remove('{}.pkl'.format(best_epoch))
            # update training variables
            best = loss_val[0]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            print('Early stop at epoch {} (no improvement in last {} epochs)'.format(epoch + 1, bad_counter))
            break

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch + 1))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    for dset in loaders.keys():
        if args.plot:
            avg_loss = test_and_plot(model, loaders[dset], loss, device, dset)
        else:
            avg_loss = test(model, loaders[dset], loss, device)
        print('Final results {}: loss = {:.6f}  MAPE {:.4f}'.format(dset, *avg_loss))

    # Nearest neighbour retrieval
    if args.closest_data_path != '':
        print("Closest string retrieval")
        closest_string_testing(encoder_model=model, data_path=args.closest_data_path,
                               batch_size=args.batch_size, device=device, distance=args.distance)

    # Hierarchical clustering
    if args.hierarchical_data_path != '':
        print("Hierarchical clustering")
        hierarchical_clustering_testing(encoder_model=model, data_path=args.hierarchical_data_path,
                                        batch_size=args.batch_size, device=device, distance=args.distance)

    # MSA tree construction on test set
    if args.construct_msa_tree == 'True':
        print("MSA tree construction")
        approximate_guide_trees(encoder_model=model, dataset=datasets['test'],
                                batch_size=args.batch_size, device=device, distance=args.distance)

    # Extra datasets testing (e.g. extrapolation)
    if args.extr_data_path != '':
        print("Extra datasets testing")
        datasets = load_edit_distance_dataset(args.extr_data_path)
        loaders = get_dataloaders(datasets, batch_size=max(1, args.batch_size // 8), workers=args.workers)

        for dset in loaders.keys():
            if args.plot:
                avg_loss = test_and_plot(model, loaders[dset], loss, device, dset)
            else:
                avg_loss = test(model, loaders[dset], loss, device)
            print('Final results {}: loss = {:.6f}  MAPE {:.4f}'.format(dset, *avg_loss))

    torch.save((model_class, model_args, model.embedding_model.state_dict(), args.distance),
               '{}.pkl'.format(model_class.__name__))


def load_edit_distance_dataset(path):
    with open(path, 'rb') as f:
        sequences, distances = pickle.load(f)

    datasets = {}
    for key in sequences.keys():
        if len(sequences[key].shape) == 2:  # datasets without batches
            if key == 'train':
                datasets[key] = EditDistanceDatasetSampled(sequences[key].unsqueeze(0), distances[key].unsqueeze(0),
                                                           multiplicity=10)
            else:
                datasets[key] = EditDistanceDatasetComplete(sequences[key], distances[key])
        else:  # datasets with batches
            datasets[key] = EditDistanceDatasetSampled(sequences[key], distances[key])
    return datasets

def load_edit_distance_dataset_calculate(path):
    with open(path, 'rb') as f:
        sequences, distances = pickle.load(f)

    datasets = {}
    for key in sequences.keys():
        if len(sequences[key].shape) == 2:  # datasets without batches
            if key == 'train':
                datasets[key] = EditDistanceDatasetSampledCalculated(sequences[key].unsqueeze(0), distances[key].unsqueeze(0),
                                                           multiplicity=10)
            else:
                datasets[key] = EditDistanceDatasetCompleteCalculated(sequences[key], distances[key])
        else:  # datasets with batches
            datasets[key] = EditDistanceDatasetSampledCalculated(sequences[key], distances[key])
    return datasets


def train(model, loader, optimizer, loss, device):
    device = 'cpu'
    avg_loss = AverageMeter()
    model.train()

    for sequences, labels in loader:
        # move examples to right device
        # sequences, labels = sequences.to(device), labels.to(device)


        with torch.autograd.set_detect_anomaly(True):
            # forward propagation
            optimizer.zero_grad()
            output = model(sequences)

            # loss and backpropagation
            loss_train = loss(output, labels)
            loss_train.backward()
            optimizer.step()

        # keep track of average loss
        avg_loss.update(loss_train.data.item(), sequences.shape[0])

    return avg_loss.avg


def test(model, loader, loss, device):
    avg_loss = AverageMeter()
    model.eval()

    for sequences, labels in loader:
        # move examples to right device
        # sequences, labels = sequences.to(device), labels.to(device)

        # forward propagation and loss computation
        output = model(sequences)
        loss_val = loss(output, labels).data.item()
        avg_loss.update(loss_val, sequences.shape[0])

    return avg_loss.avg


def test_and_plot(model, loader, loss, device, dataset):
    avg_loss = AverageMeter(len_tuple=2)
    model.eval()

    output_list = []
    labels_list = []

    for sequences, labels in loader:
        # move examples to right device
        sequences, labels = sequences.to(device), labels.to(device)

        # forward propagation and loss computation
        output = model(sequences)
        loss_val = loss[dt](output, labels).data.item()
        mape = MAPE(output, labels).data.item()
        avg_loss.update((loss_val, mape), sequences.shape[0])

        # append real and predicted distances to lists
        output_list.append(output.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())

    # save real and predicted distances for offline plotting
    outputs = np.concatenate(output_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    pickle.dump((outputs, labels), open(dataset + ".pkl", "wb"))
    # plt.plot(outputs, labels, 'o', color='black')
    # plt.show()

    return avg_loss.avg

#%%
#  Train my models
import os
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import torch
from torch import nn
import torch.optim as optim
import time
import argparse

from edit_distance.task.dataset_generator_genomic import EditDistanceGenomicDatasetGenerator
from util.data_handling.data_loader import get_dataloaders
from edit_distance.train import load_edit_distance_dataset,train,test
from edit_distance.models.pair_encoder import PairEmbeddingDistance

class LinearEncoder(nn.Module):
    """  Linear model which simply flattens the sequence and applies a linear transformation. """

    def __init__(self, len_sequence, embedding_size, alphabet_size=4):
        super(LinearEncoder, self).__init__()
        self.encoder = nn.Linear(in_features=alphabet_size * len_sequence, 
                                 out_features=embedding_size)

    def forward(self, sequence):
        # flatten sequence and apply layer
        B = sequence.shape[0]
        sequence = sequence.reshape(B, -1)
        emb = self.encoder(sequence)
        return emb

def run_model(dataset_name, embedding_size, dist_type, string_size, n_epoch):
    device = 'cpu'
    torch.manual_seed(2021)
    if device == 'cuda':
        torch.cuda.manual_seed(2021)

    # load data
    datasets = load_edit_distance_dataset(dataset_name)
    loaders = get_dataloaders(datasets, batch_size=128, workers=0)

    # model, optimizer and loss

    encoder = LinearEncoder(string_size, embedding_size)

    model = PairEmbeddingDistance(embedding_model=encoder, distance=dist_type,scaling=True)
    loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad() 


    # training
    for epoch in range(0, n_epoch):
        t = time.time()
        loss_train = train(model, loaders['train'], optimizer, loss, device)
        loss_val = test(model, loaders['val'], loss, device)

        # print progress
        
        if epoch % 5 == 0:
            print('Epoch: {:02d}'.format(epoch),
                'loss_train: {:.6f}'.format(loss_train),
                'loss_val: '.format(loss_val),
                'time: {:.4f}s'.format(time.time() - t))
        
    # testing
    for dset in loaders.keys():
        avg_loss = test(model, loaders[dset], loss, device)
        print('Final results {}: loss = {:.6f}'.format(dset, avg_loss))

    return model, avg_loss

def create_parser(out, source, train,val,test):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=out, help='Output data path')
    parser.add_argument('--train_size', type=int, default=train, help='Training sequences')
    parser.add_argument('--val_size', type=int, default=val, help='Validation sequences')
    parser.add_argument('--test_size', type=int, default=test, help='Test sequences')
    parser.add_argument('--source_sequences', type=str, default=source, help='Sequences data path')
    return parser


def generate_datasets(parser):
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

    return strings

string_size=153
n_epoch = 10
e_size=np.logspace(1,9,num=9-1, base=2,endpoint=False, dtype=int)

# dist_types=['hyperbolic', 'euclidean', 'square', 'manhattan', 'cosine']
dist_types = ['hyperbolic']

model, avg_loss = np.zeros((len(dist_types),len(e_size)),dtype=object),np.zeros((len(dist_types),len(e_size)))

names = ['largest_group_strings', 'strings_test', 'strings_subset','clean_strings']

for name in names:
    dataset_name = 'D:\hyperbolicEmbeddings' + name+'.pkl'

    for i in range(len(dist_types)):
        for j in range(len(e_size)):

            print(dist_types[i])

            model[i][j], avg_loss[i][j] = run_model(dataset_name,e_size[j],dist_types[i],string_size,n_epoch)

    pickle.dump((model,avg_loss,e_size,dist_types), open('D:\hyperbolicEmbeddings'+name+'.pkl', "wb"))


# CLUSTERING BEGINS HERE
'''
Get the hyperbolic distances between models
'''

pickle_off  = open("D:\hyperbolicEmbeddings\strings_test.pkl", "rb")
testStrings = pickle.load(pickle_off)

print(testStrings)