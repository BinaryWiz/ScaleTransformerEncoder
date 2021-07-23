# Scaling Transformer Encoder

Implementation (kind of) of a Transformer Encoder. Able to down-scale dmodel to make the dimensions smaller for later in a model. Pretty simple.

## Example
```python
import torch
from scale_transformer_encoder import ScalingLayer
x = torch.randn(16, 40, 256)
scale = ScalingLayer(in_features=256,
                     out_features=512,
                     pwff_inner_features=1028,
                     heads=8,
                     multihead_scale=False,
                     head_scale=False,
                     return_attn=True)
out, attn = scale(x)
print("Input size: {}".format(x.size()))
print("Output size: {}".format(out.size()))
print("Attention size: {}".format(attn.size()))
```

## Output
```python
Input size: torch.Size([16, 40, 768])
Output size: torch.Size([16, 40, 768])
Attention size: torch.Size([16, 8, 40, 40])
```