'''
Transformer model
'''
import torch
from torch import nn


class TransformerModel(nn.Module):

    def __init__(self, nmodal, nclass, nfeatures, nhead, nhid, nlayers, dropout=0.1, activation='relu'):
        """
        nmodal: number of different modalities (3 in this case)
        nclass: number of possible classes for output (2 for binary classification)
        nfeatures: Number of features (100 in this case)
        nhead: number of self-attention heads
        nhid: size of hidden layer for transformer
        nlayers: number of transformer layers (both encoder and decoder)
        dropout: dropout rate
        activation: activation function
        """
        super(TransformerModel, self).__init__()

        # Try using transformer for both encoder and decoder, then just put linear head on top
        self.transformer = nn.Transformer(
            nfeatures, nhead, nlayers, nlayers, nhid, dropout, activation)
        self.head = nn.Sequential(
            nn.Linear(nfeatures, nclass)
        )

        self.nmodal = nmodal

    def forward(self, src):
        """src (batch_size, nmodal, nfeatures) --> output (batch_size, nclass)"""
        compound = torch.unsqueeze(src[:, 0, :], 0)
        if self.nmodal == 3:
            crispr = torch.unsqueeze(src[:, 1, :], 0)
            orf = torch.unsqueeze(src[:, 2, :], 0)
            gene = torch.cat((crispr, orf), 0)
        else:
            gene = torch.unsqueeze(src[:, 1, :], 0)

        output = self.transformer(gene, compound)
        output = torch.squeeze(output)
        output = self.head(output)
        return output
