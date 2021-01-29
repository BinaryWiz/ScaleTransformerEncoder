import torch
import sys
from scaling_layer import ScalingLayer

src = torch.rand(32, 40, 768)
scale = ScalingLayer(768, 512, 2048, 0.3)
print(scale(src).size())
