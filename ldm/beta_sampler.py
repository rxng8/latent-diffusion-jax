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

class BetaSampler():
  def __init__(self, steps: int, beta_start: float=0.0001, beta_final: float=0.02, mode="linear") -> None:
    self._betas = np.linspace(beta_start, beta_final, steps) # (T,)
    self._alphas = 1 - self._betas # (T,)
    self._alpha_bars = np.cumprod(self._alphas) # (T,)
    self._steps = steps
    self._mode = mode
    self.fulldata = self._full()
    self.T = np.arange(self._steps)

  def __call__(self, t: np.ndarray):
    return {
      "t": t, # (B,)
      **self._sample(t), # (B,)
      **self.fulldata.copy(), # (T, B)
      "T": self.T # (T, B)
    }

  def _sample(self, t: np.ndarray):
    # t: (B,)
    alpha_bar_t = self._alpha_bars[np.asarray(t).astype(np.int32)] # (B,)
    alpha_t = np.take(self._alphas, t)
    sigma_t = np.sqrt(np.take(self._betas, t))
    return {
      "alpha_bar_t": alpha_bar_t,
      "alpha_t": alpha_t,
      "sigma_t": sigma_t,
    }
  
  def _full(self):
    li = []
    for t in range(0, self._steps):
      li.append(self._sample(t))
    return {k[:-2]: np.stack([li[i][k] for i in range(self._steps)]) for k in li[0].keys()} # (T, B)