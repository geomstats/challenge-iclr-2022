import torch.nn as nn

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