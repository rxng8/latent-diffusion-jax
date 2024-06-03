# """
# File: nets.py
# Author: Viet Nguyen
# Date: 2024-06-01

# Description: Cycle GAN components
# """

# import numpy as np
# import jax
# import jax.numpy as jnp

# from functools import partial as bind

# import embodied
# from embodied.nn import ninjax as nj
# from embodied import nn
# from embodied.nn import sg


# class Generator(nj.Module):

#   block: int = 2
#   hidden: int = 16
#   stage: int = 2

#   def __init__(self, **kw) -> None:
#     # NOTE the number of hidden is capped at 512
#     self._kw = kw

#   def __call__(self, inputs: jax.Array):
#     # inputs: (B, H, W, C)
#     B, H, W, C = inputs.shape
#     # Initial convolution layers
#     x = nn.reflection_pad_2d(inputs, 3)
#     x = self.get("in", nn.Conv2D, self.hidden, 7, stride=1, pad='valid', **self._kw)(x)
#     # encoding
#     _hidden = self.hidden * 2
#     for s in range(self.stage):
#       x = self.get(f"ds{s}", nn.Conv2D, np.minimum(_hidden, 512), 3, stride=2, pad='same', **self._kw)(x)
#       _hidden *= 2
#     # residual blocks
#     for b in range(self.block):
#       x = self.get(f"b{b}", nn.ResidualBlock)(x)
#     # upsampling, decoding
#     _hidden //= 2
#     for s in range(self.stage):
#       x = self.get(f"us{s}", nn.Conv2D, np.minimum(_hidden, 512), 3, stride=2, transp=True, pad='same', **self._kw)(x)
#     # conv out
#     # Initial convolution layers
#     kw = {**self._kw, 'act': 'tanh', 'norm': 'none'}
#     x = nn.reflection_pad_2d(x, 3)
#     x = self.get("out", nn.Conv2D, C, 7, stride=1, pad='valid', **kw)(x)
#     # return
#     return x


# class Discriminator(nj.Module):

#   stage: int = 2
#   hidden: int = 16
#   act: str = 'leaky_relu'
#   norm: str = 'instance'

#   def __init__(self, **kw) -> None:
#     # NOTE: the number of hidden is capped at 512
#     self._kw = kw
#     unused_keys = ["act", "pad", "stride", "transp", "hidden", "norm"]
#     for uk in unused_keys:
#       self._kw.pop(uk, None)

#   def output_shape(self, shape: tuple):
#     H, W = shape
#     return (H // 2**self.stage, W // 2**self.stage)

#   def __call__(self, inputs: jax.Array):
#     # inputs: (B, H, W, C)
#     B, H, W, C = inputs.shape

#     # input:
#     x = self.get("in", nn.Conv2D, self.hidden, 3, pad='same', stride=2,
#       norm='none', act=self.act, **self._kw)(inputs)
#     # downblock
#     _hidden = self.hidden * 2
#     for s in range(self.stage - 1):
#       x = self.get(f"s{s}", nn.Conv2D, np.minimum(_hidden, 512), 3, pad='same', stride=2,
#         norm=self.norm, act=self.act, **self._kw)(x)
#       _hidden *= 2
#     # out
#     x = self.get(f"out", nn.Conv2D, 1, 3, pad='same', stride=1, act='none')(x) # NOTE: TODO: do zeropadding for only the bottom right
#     return x


