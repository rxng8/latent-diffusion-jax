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


def image_grid(batch: jax.Array):
  # batch: B, H, W, C
  _batch = batch[:4]
  _, H, W, C = _batch.shape
  _batch = _batch.reshape((2, 2, H, W, C))
  _batch = _batch.transpose([0, 2, 1, 3, 4])
  _batch = _batch.reshape((2*H, 2, W, C)).reshape((2*H, 2*W, C))
  return _batch


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
        raise NotImplementedError("should all be images")
      result[key] = value
    return result

  def train(self, data):
    opt_metrics, (outs, loss_metrics) = self.opt(self.modules,
      self.loss, data, has_aux=True)
    opt_metrics.update(loss_metrics)
    return outs, opt_metrics

  def infer(self, data):
    return self.diffuser.reverse(data["image"])
  
  def report(self, data):
    x_0, xs = self.diffuser.reverse(data["image"])
    mets = {}
    mets['final'] = x_0
    #TODO: implement this


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
    scaled = {k: self.config.get(k, 1.0) * v.mean() for k, v in losses.items()}
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



class CycleGAN(nj.Module):

  def __init__(self, config: embodied.Config) -> None:
    self.config = config
    self.G_AB = Generator(**config.generator, name="G_AB")
    self.G_BA = Generator(**config.generator, name="G_BA")
    self.D_A = Discriminator(**config.discriminator, name="D_A")
    self.D_B = Discriminator(**config.discriminator, name="D_B")
    self.opt = nn.Optimizer(**config.opt, name="opt")
    self.modules = [self.G_AB, self.G_BA, self.D_A, self.D_B]
    self._image_shape = config.image_size[::-1] # [W, H] -> [H, W]
    self._dis_out = self.D_A.output_shape(self._image_shape)

  def train(self, data: dict):
    mets = {}
    opt_mets, (out, loss_mets) = self.opt(self.modules, self.loss, self.preprocess(data), has_aux=True)
    mets.update(opt_mets)
    mets.update(loss_mets)
    return out, mets

  def preprocess(self, data):
    result = {}
    for key, value in data.items():
      if len(value.shape) >= 3 and value.dtype == jnp.uint8:
        value = (nn.cast(value) / 255.0 - 0.5) * 2
      else:
        raise NotImplementedError("should all be images")
      result[key] = value
    return result

  def loss(self, data: dict) -> tuple:
    real_A = data['A']
    real_B = data['B']
    B, H, W, C = real_A.shape
    valid = jnp.ones((B, *self._dis_out, 1))
    fake = jnp.zeros((B, *self._dis_out, 1))

    losses = {}
    # identity loss
    id_A = jnp.abs(self.G_BA(real_A) - real_A) # If put A into G:B->A. Of course it still return A
    id_B = jnp.abs(self.G_AB(real_B) - real_B) # If put B into G:A->B. Of course it still return B
    losses["id"] = (id_A + id_B) / 2

    # GAN loss
    fake_B = self.G_AB(real_A)
    fake_A = self.G_BA(real_B)
    gan_AB = (self.D_B(fake_B) - valid)**2 # trick discriminator B
    gan_BA = (self.D_A(fake_A) - valid)**2 # trick discriminator A
    losses["gan"] = (gan_AB + gan_BA) / 2
    
    # Cycle loss
    recov_A = self.G_BA(fake_B)
    recov_B = self.G_AB(fake_A)
    cycle_A = jnp.abs(real_A - recov_A)
    cycle_B = jnp.abs(real_B - recov_B)
    losses["cycle"] = (cycle_A + cycle_B) / 2

    # discriminator loss
    disc_real_A = (self.D_A(real_A) - valid)**2
    disc_fake_A = (self.D_A(sg(fake_A)) - fake)**2
    disc_real_B = (self.D_B(real_B) - valid)**2
    disc_fake_B = (self.D_B(sg(fake_B)) - fake)**2
    losses["disc"] = (disc_real_A + disc_real_B + disc_fake_A + disc_fake_B) / 4

    # scaled and compute total losses
    scaled = {k: self.config.loss_scales[k] * v for k, v in losses.items()}
    model_loss = sum([x.mean() for x in scaled.values()])
    mets = self._metrics(data, losses)
    outs = {
      "real_A": real_A,
      "fake_A": fake_A,
      "real_B": real_B,
      "fake_B": fake_B,
    }
    return model_loss, (outs, mets)

  def _metrics(self, data: dict, losses: dict):
    mets = {}
    for k, v in losses.items():
      mets.update(nn.tensorstats(v, prefix=k))
    return mets

  def report(self, data):
    _, (outs, mets) = self.loss(self.preprocess(data))
    B = list(outs.values())[0].shape[0]
    for i in range(max(B, 4)):
      img = jnp.stack([outs["real_A"][i], outs["fake_B"][i], outs["real_B"][i], outs["fake_A"][i]], axis=0)
      mets.update({f"image/batch_{i}": image_grid(img / 2 + 0.5)})
    return mets

