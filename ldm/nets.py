"""
File: nets.py
Author: Viet Nguyen
Date: 2024-06-03

Description: all networks logic stays here
"""

import numpy as np
import jax
import jax.numpy as jnp

from functools import partial as bind

import embodied
from embodied.nn import ninjax as nj
from embodied import nn
from embodied.nn import sg



class NoiseEstimatorUNet(nj.Module):

  stage: int = 3
  head: int = 8
  group: int = 8
  thidden: int = 512
  ahidden: int = 128
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
    # dim = self._hidden * 2
    dim = self._hidden
    for s in range(self.stage):
      # two res blocks + self attention + add together
      x = self.get(f"dr{s}0", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"dr{s}1", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      # x = self.get(f"da{s}", nn.CrossAttentionBlock, self.ahidden, head=self.head, group=self.group)(x, condition)
      carries.append(x)
      # Downsampling
      if s != self.stage - 1:
        x = self.get(f"dd{s}", nn.Conv2D, dim, 3, stride=2)(x)
      # increase dimension
      # dim *= 2

    #### bottleneck
    x = self.get(f"br0", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
    x = self.get(f"ba", nn.CrossAttentionBlock, self.ahidden, head=self.head, group=self.group)(x, condition)
    x = self.get(f"br1", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)

    #### upsampling
    # Upsampling phase
    # dim //= 2
    for s in reversed(range(self.stage)):
      x = jnp.concatenate([carries.pop(), x], -1)
      # two res blocks + self attention + add together
      x = self.get(f"ur{s}0", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"ur{s}1", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      # x = self.get(f"ua{s}", nn.CrossAttentionBlock, self.ahidden, head=self.head, group=self.group)(x, condition)
      # Upsampling
      if s != 0:
        x = self.get(f"uu{s}", nn.Conv2D, dim, 3, stride=2, transp=True)(x)
      # decrease dimension
      # dim //= 2

    # Final ResNet block and output convolutional layer
    x = self.get(f"fr", nn.ResidualTimeBlock, self._hidden, act=self.act, group=self.group)(x, time_embed)
    x = self.get(f"out", nn.Conv2D, C, 1, stride=1)(x)
    return x



class Diffuser(nj.Module):

  hidden: int = 128
  stage: int = 3
  head: int = 8
  group: int = 8
  thidden: int = 512
  ahidden: int = 128
  act: str = "silu"

  # implement the algorithm from https://arxiv.org/pdf/2006.11239.pdf
  # adapt from: https://github.com/andylolu2/jax-diffusion/blob/main/jax_diffusion/diffusion.py
  def __init__(self, beta_start: float, beta_final: float, steps: int):
    """_summary_

    Args:
      beta_start (float): the beta/variance of x_0 or the observation/frame/image
      beta_final (float): the beta/variance of x_T-1 or the latent/diffused noise
      steps (int): the total number of steps in the mdp
    """
    self._betas = np.linspace(beta_start, beta_final, steps) # (T,)
    self._alphas = 1 - self._betas # (T,)
    self._alpha_bars = np.cumprod(self._alphas) # (T,)
    self._steps = steps # ()
    self.unet = NoiseEstimatorUNet(self.hidden, stage = self.stage, head = self.head,
      group = self.group, thidden = self.thidden, ahidden=self.ahidden, act = self.act, name="unet")

  def forward(self, x_0: jax.Array, t: jax.Array) -> jax.Array:
    # given the image, add noise to it. See algorithm 1 in https://arxiv.org/pdf/2006.11239.pdf
    # For forward, we can compute a random t
    B, H, W, C = x_0.shape
    (B,) = t.shape
    """x_t, eps = self.sample_q(x_0, t): Samples x_t given x_0 by the q(x_t|x_0) formula."""
    # x_0: (B, H, W, C)
    alpha_bar_t = self._alpha_bars.take(t.astype(jnp.int32)) # (B,)
    alpha_bar_t = alpha_bar_t[:, None, None, None] # (B, 1, 1, 1)
    eps = jax.random.normal(nj.seed(), shape=x_0.shape, dtype=x_0.dtype)
    x_t = jnp.sqrt(alpha_bar_t) * x_0 + jnp.sqrt(1 - alpha_bar_t) * eps
    """end of x_t, eps = self.sample_q(x_0, t)"""
    # x_t = x_t.clip(-1, 1)
    return x_t, eps

  def reverse_step(self, x_t: jax.Array, xs: tuple):
    t, cond = xs # (B,), (B,)
    """See algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""
    B, H, W, C = x_t.shape
    (B,) = t.shape
    alpha_t = jnp.take(self._alphas, t)[:, None, None, None] # (B,) -> (B, H, W, C)
    alpha_bar_t = jnp.take(self._alpha_bars, t)[:, None, None, None] # (B,) -> (B, H, W, C)
    sigma_t = jnp.sqrt(jnp.take(self._betas, t))[:, None, None, None] # (B,) -> (B, H, W, C)
    z = (t > 0)[:, None, None, None] * jax.random.normal(nj.seed(), shape=x_t.shape, dtype=x_t.dtype)
    eps = self.unet(x_t, t, cond[:, None, None])
    x = (1.0 / jnp.sqrt(alpha_t)) * (
      x_t - ((1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t)) * eps
    ) + sigma_t * z
    # x = x.clip(-1, 1)
    return x, x

  def reverse(self, x_T: jax.Array, cond: jax.Array) -> jax.Array:
    # (B, H, W, C) -> (B,) => (B, H, W, C) -> (T, B, H, W, C)
    B, H, W, C = x_T.shape
    # given the noise, reconstruct the image
    # For reverse, we have to reverse it one by one
    ts = jnp.arange(0, self._steps)[:, None] # (T, 1)
    ts = jnp.repeat(ts, B, axis=1) # (T, B)
    conds = jnp.repeat(cond[None], self._steps, axis=0)
    xs = (ts, conds)
    x_hat_0, xs = nj.scan(self.reverse_step, x_T, xs, reverse=True, unroll=1, axis=0)
    return x_hat_0, xs
  

