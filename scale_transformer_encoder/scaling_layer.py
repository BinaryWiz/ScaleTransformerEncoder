import torch
import torch.nn as nn
from scale_transformer_encoder.multihead import MultiHeadSelfAttn
from scale_transformer_encoder.position_wise_feed_forward import PositionWiseFeedForward

class ScalingLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 pwff_inner_features,
                 heads=8,
                 multihead_scale=False,
                 head_scale=False,
                 return_attn=False, 
                 pwff_dropout=0.1):
        '''
        A transformer that can have a different input and output dimension.
        in_features: The size of the input dimension
        out_features: The size of the output dimension
        pwff_inner_features: The size of the position-wise feed forward network (usually larger than output dimension)
        heads: Number of heads in the multi-head self attention.
        multihead_scale: Set to true if the scaling from the input dimension to the output dimension should happen 
                         in the multi-head module. By default, it happens in the position-wise feed forward network.
        head_scale: Only applies if multihead_scale is set to true. By default, the scaling in the multi-head will happen
                    in the last linear layer. Set to true if it should happen in each head of the module.
                    Check multihead.py for more information.
        return_attn: Default is false. Set to true if the transformer should also output the attention matrix.
        pwff_dropout: Set the dropout for the position-wise feed forward network.
        '''
        
        super(ScalingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pwff_inner_features = pwff_inner_features
        self.heads = heads
        self.multihead_scale = multihead_scale
        self.head_scale = head_scale,
        self.return_attn = return_attn
        self.pwff_dropout = pwff_dropout
        
        if self.multihead_scale:
            # Multi-Head Self Attention (scaling)
            self.multihead = MultiHeadSelfAttn(in_features=self.in_features,
                                            out_features=self.out_features,
                                            heads=self.heads,
                                            head_scale=self.head_scale)
            
            # Position-Wise Feed Forward (no scaling)
            self.pwff = PositionWiseFeedForward(in_features=self.out_features,
                                            inner_features=self.pwff_inner_features,
                                            out_features=self.out_features,
                                            dropout=self.pwff_dropout)
        
        else:
            # Multi-Head Self Attention (no scaling)
            self.multihead = MultiHeadSelfAttn(in_features=self.in_features,
                                            out_features=self.in_features,
                                            heads=self.heads,
                                            head_scale=False)

            # Position-Wise Feed Forward (scaling)
            self.pwff = PositionWiseFeedForward(in_features=self.in_features,
                                                inner_features=self.pwff_inner_features,
                                                out_features=self.out_features,
                                                dropout=self.pwff_dropout)
        
        # This is used to scale the original embedding to make a residual connection if in_features != out_features
        if self.in_features != self.out_features:
            self.residual_scale = nn.Linear(in_features=self.in_features,
                                            out_features=self.out_features)
        
        # The Layer Normalization layers
        if self.multihead_scale:
            self.multihead_ln = nn.LayerNorm(self.out_features)
        else:
            self.multihead_ln = nn.LayerNorm(self.in_features)

        self.pwff_ln = nn.LayerNorm(self.out_features)
    
    def forward(self, embeddings):
        # This will be for adding later
        residual = embeddings.clone()
        
        # Forward propagate through the multihead attention layer
        out, attn = self.multihead(embeddings)
        
        # Scale the original embeddings down to add element-wise (if in_features != out_features)
        if self.multihead_scale and (self.in_features != self.out_features):
            residual = self.residual_scale(residual)

        # Layer normalization            
        out = self.multihead_ln(out + residual)
        
        # Keep a copy of out now to add after the Position-Wise Feed Forward
        residual = out.clone()
        
        # Forward propagate through the Position-Wise Feed Forward
        out = self.pwff(out)
        
        # Scale the residual if the scaling happens in PWFF
        if not self.multihead_scale and (self.in_features != self.out_features):
            residual = self.residual_scale(residual)

        # Complete the residual connection and apply Layer Normalization
        out = self.pwff_ln(out + residual)

        if self.return_attn:
            return out, attn

        return out

if __name__ == "__main__":
    x = torch.randn(16, 40, 768)
    scale = ScalingLayer(768, 768, 2048, multihead_scale=False, head_scale=False, return_attn=True)
    out, attn = scale(x)
    print("Input size: {}".format(x.size()))
    print("Output size: {}".format(out.size()))
    print("Attention size: {}".format(attn.size()))