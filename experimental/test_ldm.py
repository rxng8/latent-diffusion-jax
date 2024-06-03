# %%

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import warnings

import numpy as np
import jax
import jax.numpy as jnp

from embodied import nn
from embodied.nn import ninjax as nj

warnings.filterwarnings('ignore', '.*input data to the valid range.*')

transform = lambda x: x / 255.0 * 2 - 1
untransform = lambda x: ((x / 2 + 0.5) * 255.0).astype(np.uint8)


class NoiseEstimatorUNet(nj.Module):

  stage: int = 3
  head: int = 1
  group: int = 1
  thidden: int = 32
  act: str = "gelu"

  def __init__(self, hidden: int) -> None:
    self._hidden = hidden

  def __call__(self, imgnoi: jax.Array, timeid: jax.Array, condition: jax.Array) -> jax.Array:
    # imgnoi (B, H, W, C), -> timeid: (B,) -> condition (B, H, W, C) or (B, S, C) -> noise (B, H, W, C)
    """_summary_

    Args:
        imgnoi (jax.Array): (B, H, W, C)
        timeid (jax.Array): (B,)
        condition (jax.Array): (B, S, Z) or (B, H, W, Z)

    Returns:
        jax.Array: (B, Z1)
    """
    B, H, W, C = imgnoi.shape
    x = self.get("conv", nn.Conv2D, self._hidden, 1)(imgnoi)
    time_embed = self.get("time", nn.TimeEmbedding, self.thidden)(timeid)
    carries = []
    #### downsampling
    dim = self._hidden * 2
    for s in range(self.stage):
      # two res blocks + self attention + add together
      x = self.get(f"dr{s}0", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"dr{s}1", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"da{s}", nn.CrossAttentionBlock, dim, head=self.head, group=self.group)(x, condition)
      carries.append(x)
      # Downsampling
      if s != self.stage - 1:
        x = self.get(f"dd{s}", nn.Conv2D, dim, 3, stride=2, act=self.act)(x)
      # increase dimension
      dim *= 2

    #### bottleneck
    x = self.get(f"br0", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
    x = self.get(f"ba", nn.CrossAttentionBlock, dim, head=self.head, group=self.group)(x, condition)
    x = self.get(f"br1", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)

    #### upsampling
    # Upsampling phase
    dim //= 2
    for s in reversed(range(self.stage)):
      x = jnp.concatenate([carries.pop(), x], -1)
      # two res blocks + self attention + add together
      x = self.get(f"ur{s}0", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"ur{s}1", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"ua{s}", nn.CrossAttentionBlock, dim, head=self.head, group=self.group)(x, condition)
      # Upsampling
      if s != 0:
        x = self.get(f"uu{s}", nn.Conv2D, dim, 3, stride=2, transp=True, act=self.act)(x)
      # decrease dimension
      dim //= 2

    # Final ResNet block and output convolutional layer
    x = self.get(f"fr", nn.ResidualTimeBlock, self._hidden, act=self.act, group=self.group)(x, time_embed)
    x = self.get(f"out", nn.Conv2D, C, 1, stride=1)(x)
    return x


N = NoiseEstimatorUNet(1, name="U")
B, H, W, C = 2, 64, 64, 3
_, S, Z = 2, 40, 16
img = jnp.asarray(np.random.normal(0, 1, (B, H, W, C)))
cond = jnp.asarray(np.random.normal(0, 1, (B, S, Z)))
tid = jnp.asarray(np.random.randint(0, 200, (B,)))

params = nj.init(N)({}, img, tid, cond, seed=0)
fn = jax.jit(nj.pure(N))
_, out = fn(params, img, tid, cond)
