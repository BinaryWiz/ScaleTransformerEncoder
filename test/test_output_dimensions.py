import torch
from scale_transformer_encoder.scaling_layer import ScalingLayer
import unittest

class TestOutDims(unittest.TestCase):
    '''
    Test various dimensions for the transformer encoder.
    '''
    
    def test_no_scale(self):
        x = torch.randn(4, 20, 128)
        scale = ScalingLayer(128, 128, 256, multihead_scale=False, head_scale=False, return_attn=False)
        out = scale(x)
        self.assertEqual(list(out.size()), [4, 20, 128])
    
    def test_scaled_pwff(self):
        x = torch.randn(4, 20, 128)
        scale = ScalingLayer(128, 256, 512, multihead_scale=False, head_scale=False, return_attn=False)
        out = scale(x)
        self.assertEqual(list(out.size()), [4, 20, 256])
    
    def test_scaled_multihead_linear(self):
        x = torch.randn(4, 20, 128)
        scale = ScalingLayer(128, 256, 512, multihead_scale=True, head_scale=False, return_attn=False)
        out = scale(x)
        self.assertEqual(list(out.size()), [4, 20, 256])
        
    def test_scaled_multihead_head(self):
        x = torch.randn(4, 20, 128)
        scale = ScalingLayer(128, 256, 512, multihead_scale=True, head_scale=True, return_attn=False)
        out = scale(x)
        self.assertEqual(list(out.size()), [4, 20, 256])

    def test_attn_and_heads(self):
        x = torch.randn(4, 20, 128)
        scale = ScalingLayer(128, 128, 256, heads=4, multihead_scale=False, head_scale=False, return_attn=True)
        _, attn = scale(x)
        self.assertEqual(list(attn.size()), [4, 4, 20, 20])

if __name__ == '__main__':
    unittest.main()