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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2021)
    if device == 'cuda':
        torch.cuda.manual_seed(2021)

    # load data
    datasets = load_edit_distance_dataset(dataset_name)
    loaders = get_dataloaders(datasets, batch_size=128, workers=5)

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
        '''
        if epoch % 5 == 0:
            print('Epoch: {:02d}'.format(epoch),
                'loss_train: {:.6f}'.format(loss_train),
                'loss_val: '.format(loss_val),
                'time: {:.4f}s'.format(time.time() - t))
        '''
        
        
    # testing
    for dset in loaders.keys():
        avg_loss = test(model, loaders[dset], loss, device)
        # print('Final results {}: loss = {:.6f}'.format(dset, avg_loss))

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
