import torch
import torch.nn as nn
from multihead import MultiHeadSelfAttn
from position_wise_feed_forward import PositionWiseFeedForward

class ScalingLayer(nn.Module):
    def __init__(self, in_features, out_features, pwff_inner_features, pwff_dropout=0.1):
        super(ScalingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pwff_inner_features = pwff_inner_features
        self.pwff_dropout = pwff_dropout

        # Multi-Head Self Attention
        self.multihead = MultiHeadSelfAttn(in_features=self.in_features,
                                           out_features=self.out_features)
        
        # Position-Wise Feed Forward
        self.pwff = PositionWiseFeedForward(in_features=self.out_features,
                                            inner_features=self.pwff_inner_features,
                                            out_features=self.out_features,
                                            dropout=self.pwff_dropout)
        
        # This is used to scale the original embedding to make a residual connection if in_features != out_features
        self.residual_scale = nn.Linear(in_features=self.in_features,
                                        out_features=self.out_features)
        
        # The Layer Normalization layers
        self.multihead_ln = nn.LayerNorm(self.out_features)
        self.pwff_ln = nn.LayerNorm(self.out_features)
    
    def forward(self, embeddings):
        # This will be for adding later
        residual = embeddings.clone()
        
        # Forward propagate through the multihead attention layer
        out = self.multihead(embeddings)
        
        # Scale the original embeddings down to add element-wise
        residual = self.residual_scale(residual)
        out = self.multihead_ln(out + residual)
        
        # Keep a copy of out now to add after the Position-Wise Feed Forward
        residual = out.clone()
        
        # Forward propagate through the Position-Wise Feed Forward
        out = self.pwff_ln(out)
        
        # Complete the residual connection and apply Layer Normalization
        out = self.pwff_ln(out + residual)
        return out
