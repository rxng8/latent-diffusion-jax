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
  def __init__(self, beta_start, beta_final, steps) -> None:
    self._betas = np.linspace(beta_start, beta_final, steps) # (T,)
    self._alphas = 1 - self._betas # (T,)
    self._alpha_bars = np.cumprod(self._alphas) # (T,)
    self._steps = steps
    self.fulldata = self._full()

  def __call__(self, t: np.ndarray):
    return {
      **self._step(t),
      **self.fulldata.copy()
    }

  def _step(self, t: np.ndarray):
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
      li.append(self._step(t))
    return {k[:-2]: np.stack([li[i][k] for i in range(self._steps)]) for k in li[0].keys()} # (T, B)