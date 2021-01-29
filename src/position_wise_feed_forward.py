import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, in_features, inner_features, out_features, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        
        # First linear layer goes from the in_features to inner_features dimension
        self.fc1 = nn.Linear(in_features, inner_features)
        
        # Second linear layer goes from inner_features to out_features
        self.fc2 = nn.Linear(inner_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embeddings):
        '''
        Forward propagate through the Feed-Forward network
        '''
        
        embeddings = self.fc1(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.relu(self.fc2(embeddings))
        embeddings = self.dropout(embeddings)
        return embeddings
