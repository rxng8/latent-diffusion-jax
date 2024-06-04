"""
File: trainer.py
Author: Viet Nguyen
Date: 2024-06-01

Description: Trainer of CycleGAN model
"""


from ruamel import yaml
import numpy as np
import jax
import jax.numpy as jnp

from functools import partial as bind

import embodied
from embodied.nn import ninjax as nj
from embodied import nn
from embodied.nn import sg

from .nets import Diffuser


def image_grid(inputs: jax.Array):
  # inputs: T, B, H, W, C
  T, B, H, W, C = inputs.shape
  b = min(B, 4)
  out = inputs[:, :b]
  out = out.transpose([1, 2, 0, 3, 4])
  out = out.reshape((b*H, T, W, C)).reshape((b*H, T*W, C))
  return out

def video_grid(inputs: jax.Array):
  T, B, H, W, C = inputs.shape
  b = int(np.sqrt(B))
  b = min(b, 4)
  out = inputs[:, :b*b]
  out = out.reshape((T, b, b, H, W, C))
  out = out.transpose([0, 1, 3, 2, 4, 5])
  out = out.reshape((T, b*H, b, W, C)).reshape((T, b*H, b*W, C))
  return out

class DiffusionTrainer(nj.Module):
  def __init__(self, config):
    self.config = config
    self.diffuser = Diffuser(**config.diffuser, name="diff")
    self.opt = nn.Optimizer(**config.opt, name="opt")
    self.modules = [self.diffuser]

  def preprocess(self, data):
    result = {}
    for key, value in data.items():
      if len(value.shape) >= 3 and value.dtype == jnp.uint8:
        value = (nn.cast(value) / 255.0 - 0.5) * 2
      else:
        raise NotImplementedError(f"should all be images! got dtype={value.dtype}, shape: {value.shape}")
      result[key] = value
    return result

  def train(self, data):
    _data = self.preprocess(data)
    opt_metrics, (outs, loss_metrics) = self.opt(self.modules, self.loss, _data, has_aux=True)
    opt_metrics.update(loss_metrics)
    return outs, opt_metrics

  def infer(self, data):
    _data = self.preprocess(data)
    return self.diffuser.reverse(_data["image"])
  
  def report(self, data):
    _data = self.preprocess(data)
    x_0, xs = self.diffuser.reverse(_data["image"]) # (B, H, W, C), (T, B, H, W, C)
    _, (outs, loss_mets) = self.loss(_data)
    mets = {}
    mets.update(loss_mets)
    mets['final'] = x_0
    idx = np.linspace(0, self.diffuser._steps - 1, 9).astype(np.int32)
    mets['image'] = image_grid((jnp.take(xs, idx, axis=0) / 2 + 0.5).clip(0, 1)) # (T->9, B->4, H, W, C)
    idx = np.linspace(0, self.diffuser._steps - 1, 20).astype(np.int32)
    mets['video'] = video_grid((jnp.take(xs, idx, axis=0) / 2 + 0.5).clip(0, 1)) # (T->9, B->4, H, W, C)
    return mets

  def loss(self, data):
    x_0 = data["image"]
    B, H, W, C = x_0.shape

    # Generate random timesteps indices
    timesteps = np.random.randint(0, self.diffuser._steps, (B,))
    timesteps = nn.cast(timesteps)

    # Generating the noise and noisy image for this batch
    # Add noise to x_0 until timestep
    noisy_image, noise = self.diffuser.forward(x_0, timesteps)

    # Forward noising: given a noisy image, predict the noise added to that image
    pred_noise = self.diffuser.unet(noisy_image, timesteps)

    # l1 loss
    # loss = ((pred_noise - noise)**2).mean([-3, -2, -1]).mean()
    losses = {}
    losses["diffuser"] = (pred_noise - noise)**2
    scaled = {k: self.config.loss_scales.get(k, 1.0) * v.mean() for k, v in losses.items()}
    model_loss = sum(scaled.values())

    # metrics
    outs = {"pred_noise": pred_noise, "noise": noise, "noisy_image": noisy_image}
    mets = self._metrics(data, losses)
    return model_loss, (outs, mets)

  def _metrics(self, data: dict, losses: dict):
    mets = {}
    for k, v in losses.items():
      mets.update(nn.tensorstats(v, prefix=f"losses/{k}"))
    return mets

