import torch
import torch.nn as nn
import math

class MultiHeadSelfAttn(nn.Module):
    def __init__(self, in_features, out_features, heads=8, head_scale=False):
        '''
        in_features: should be equal to the embedding_dimension
        output: what the size of the output embedding should be
        '''
        
        super(MultiHeadSelfAttn, self).__init__()
        self.in_features = in_features
        self.heads = heads
        self.out_features = out_features
        self.head_features = in_features
        
        if head_scale:
            self.head_features = self.out_features
        
        # Make sure that in_features is compatible with the number of heads
        assert self.head_features % heads == 0
        
        # dk is the size of each of the linear projections of the embedding
        self.dk = self.head_features // 8
        
        # These are the parameters to project the matrix to the amount of heads
        self.key_projections = nn.Linear(self.in_features, self.head_features)
        self.value_projections = nn.Linear(self.in_features, self.head_features)
        self.query_projections = nn.Linear(self.in_features, self.head_features)
        
        # The final linear layer
        self.end_linear = nn.Linear(self.head_features, self.out_features)
        
        # Softmax
        self.softmax = nn.Softmax(dim=2)
        
    def scaled_attention(self, head_query, head_keys, head_values):
        # Calculate the scaled dot-product
        attn = torch.matmul(head_query, head_keys.transpose(-2, -1)) / math.sqrt(self.dk)
        
        # Get the softmax
        attn = self.softmax(attn)
        
        # Multiply the softmax output by the values
        attn_head = torch.matmul(attn, head_values)
        return attn_head, attn
        
    def forward(self, embeddings):
        '''
        Forward propagate through the multi-head attention
        embeddings: should be of dimensions (batch, sequence_length, embedding_dimension)
        '''
        
        batches, sequence_length, embeddings_dim = embeddings.size()
        
        # Get the query projections
        query = self.query_projections(embeddings)
        query = query.view(batches, self.heads, sequence_length, self.dk)
        
        # Get the key projections
        keys = self.key_projections(embeddings)
        keys = keys.view(batches, self.heads, sequence_length, self.dk)
        
        # Get the value projections
        values = self.value_projections(embeddings)
        values = values.view(batches, self.heads, sequence_length, self.dk)
        
        # Calculated the scaled dot-product attention
        attn_head, attn = self.scaled_attention(query, keys, values)
        
        # Put it in dimensions (batches, sequence_length, in_features)
        attn_head = attn_head.view(batches, sequence_length, self.head_features)
        
        # Apply the final linear layer
        return self.end_linear(attn_head), attn


if __name__ == "__main__":
    x = torch.randn(1, 5, 16)
    multi = MultiHeadSelfAttn(16, 16)
    out, attn = multi(x)
    print("Input size: {}".format(x.size()))
    print("Output size: {}".format(out.size()))
    print("Attention size: {}".format(attn.size()))
