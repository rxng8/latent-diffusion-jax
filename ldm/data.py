"""
File: data.py
Author: Viet Nguyen
Date: 2024-06-04

Description: Data handling
"""

from typing import List, Dict
from abc import ABC, abstractmethod
import sys, pathlib
import numpy as np
import jax
import jax.numpy as jnp
import os
from PIL import Image
import cv2

import embodied
from .beta_sampler import BetaSampler

class Dataloader(ABC):
  def __init__(self, config, mode="train") -> None:
    self._batch_size = config.batch_size
    self._height = config.image_size[1]
    self._width = config.image_size[0]
    self._mode = mode
    self.config = config
    self.beta_sampler = BetaSampler(config.diffuser_steps, **config.beta_sampler)

  def sample(self):
    raise NotImplementedError(f"() -> dict: ('image': (B, H, W, C), 'class': (B,)). Image uint8 [0, 255]")

  def dataset(self):
    """Given the `_sample` method, we yield data returned together with the sampled beta
      from the beta sampler.

    Yields:
      dict: {
        # handled by subclasses
        "image": (B, H, W, C): the training image
        "class": (B,): the condition class
        # handled by this parent class
        "t": (B,): the sampled mdp time step
        "alpha_t": (B,): the corresponding alpha at sampled time step
        "alpha_bar_t": (B,): the corresponding alpha bar at sampled time step
        "sigma_t": (B,): the corresponding sigma at sampled time step
        "T": (T,): the whole time step from 0 to max diffusion time steps
        "alpha": (T,): the whole alpha array through mdp time steps
        "alpha_bar": (T,): the whole alpha bar array through mdp time steps
        "sigma": (T,): the whole sigma array through mdp time steps
      }
    """
    while True:
      t = np.random.randint(0, self.config.diffuser_steps, size=(self._batch_size,))
      yield {**self.sample(), **self.beta_sampler(t)}

class ElefantDataset:
  def __init__(self, config, mode="train") -> None:
    self._batch_size = config.batch_size
    elefant = pathlib.Path(__file__).parent / "elefant.jpg"
    elefant = Image.open(elefant)
    elefant = elefant.resize(config.image_size)
    elefant = np.asarray(elefant)[None]
    elefant = np.repeat(elefant, config.batch_size, 0) # (B, H, W, C)
    self.elefant = {"image": elefant, "class": np.zeros((config.batch_size))}
    self._steps = config.diffuser_steps
    self.beta_sampler = BetaSampler(config.diffuser_steps, **config.beta_sampler)
    self._mode = mode

  def _sample(self):
    t = np.random.randint(0, self._steps, size=(self._batch_size,))
    data = {**self.elefant, **self.beta_sampler(t)}
    return data

  def dataset(self):
    while True:
      yield self._sample()


class SingleDomainDataset:
  def __init__(self, config, mode="train") -> None:
    self.path_A = pathlib.Path(config.dir_path).resolve()
    print(f"loading 1 domains from {self.path_A}")
    image_size = config.image_size
    self.H = image_size[1]
    self.W = image_size[0]
    self._batch_size = config.batch_size
    self.epoch_cnt = 0
    self._steps = config.diffuser_steps
    self.beta_sampler = BetaSampler(config.diffuser_steps, **config.beta_sampler)
    self._n_classes = config.n_classes

    # get all image path
    self.domain_A = []
    for fname in os.listdir(self.path_A):
      if self._check_image(fname):
        self.domain_A.append(self.path_A / fname)
    self._len_A = len(self.domain_A)

  def _check_image(self, fname: str):
    accepted = [".png", ".jpg", ".jpeg"]
    for a in accepted:
      if fname.endswith(a):
        return True
    return False

  def _sample_one(self, iA):
    # iA = np.random.randint(0, self._len_A)
    img_A = np.asarray(Image.open(self.domain_A[iA]))
    img_A = np.asarray(cv2.resize(img_A, (self.W, self.H))) if (self.W, self.H) != img_A.shape[:2] else img_A
    return {"image": img_A}

  def _sample(self, idx, current):
    batch = []
    for i in range(self._batch_size):
      data = self._sample_one(idx[(current + i) % len(self.domain_A)])
      batch.append(data)
    data = {k: np.stack([batch[i][k] for i in range(self._batch_size)], 0) for k in batch[0].keys()}
    t = np.random.randint(0, self._steps, (self._batch_size,))
    return {**data, **self.beta_sampler(t), "class": np.zeros((self._batch_size,))}

  def dataset(self):
    idx = np.random.permutation(np.arange(0, len(self.domain_A)))
    current = 0
    while True:
      out = self._sample(idx, current)
      current += self._batch_size
      if current >= len(self.domain_A):
        current = 0
        idx = np.random.permutation(np.arange(0, len(self.domain_A)))
        self.epoch_cnt += 1
      yield out

class TwoDomainDataset:
  def __init__(self, config, mode="train") -> None:
    self.path_A = pathlib.Path(config.path_A).resolve()
    self.path_B = pathlib.Path(config.path_B).resolve()
    print(f"loading 1 domains from {self.path_A}")
    print(f"loading 1 domains from {self.path_B}")
    image_size = config.image_size
    self.H = image_size[1]
    self.W = image_size[0]
    self._batch_size = config.batch_size
    self.epoch_cnt = 0
    self._steps = config.diffuser_steps

    # get all image path
    self.domain_A = []
    for fname in os.listdir(self.path_A):
      if self._check_image(fname):
        self.domain_A.append(self.path_A / fname)
    self._len_A = len(self.domain_A)

    self.domain_B = []
    for fname in os.listdir(self.path_B):
      if self._check_image(fname):
        self.domain_B.append(self.path_B / fname)
    self._len_B = len(self.domain_B)

    self.beta_sampler = BetaSampler(config.diffuser_steps, **config.beta_sampler)

  def _check_image(self, fname: str):
    accepted = [".png", ".jpg", ".jpeg"]
    for a in accepted:
      if fname.endswith(a):
        return True
    return False

  def _sample_one_A(self, iA):
    # iA = np.random.randint(0, self._len_A)
    img_A = np.asarray(Image.open(self.domain_A[iA]))
    img_A = np.asarray(cv2.resize(img_A, (self.W, self.H))) if (self.W, self.H) != img_A.shape[:2] else img_A
    return {"image": img_A, "class": np.zeros(())}
  
  def _sample_one_B(self, iB):
    img_B = np.asarray(Image.open(self.domain_B[iB]))
    img_B = np.asarray(cv2.resize(img_B, (self.W, self.H))) if (self.W, self.H) != img_B.shape[:2] else img_B
    return {"image": img_B, "class": np.ones(())}

  def _sample(self, idx, current):
    batch = []
    for i in range(self._batch_size):
      id = idx[(current + i) % (len(self.domain_A) + len(self.domain_B))]
      if id < self._len_A:
        data = self._sample_one_A(id)
      else:
        data = self._sample_one_B(id - self._len_A)
      batch.append(data)
    data = {k: np.stack([batch[i][k] for i in range(self._batch_size)], 0) for k in batch[0].keys()}
    t = np.random.randint(0, self._steps, (self._batch_size,))
    return {**data, **self.beta_sampler(t)}

  def dataset(self):
    idx = np.random.permutation(np.arange(0, len(self.domain_A) + len(self.domain_B)))
    current = 0
    while True:
      out = self._sample(idx, current)
      current += self._batch_size
      if current >= len(self.domain_A):
        current = 0
        idx = np.random.permutation(np.arange(0, len(self.domain_A)))
        self.epoch_cnt += 1
      yield out

class RollingDataset:
  def __init__(self, config, mode="train") -> None:
    self.path_A = (pathlib.Path(__file__).parent / "permanent_data").resolve()
    image_size = config.image_size
    self.H = image_size[1]
    self.W = image_size[0]
    self._batch_size = config.batch_size
    self.epoch_cnt = 0
    self._steps = config.diffuser_steps
    self.beta_sampler = BetaSampler(config.diffuser_steps, **config.beta_sampler)
    self._n_classes = config.n_classes

    # get all image path
    self.domain_A = []
    for fname in os.listdir(self.path_A):
      if self._check_image(fname):
        self.domain_A.append(self.path_A / fname)
    self._len_A = len(self.domain_A)

  def _check_image(self, fname: str):
    accepted = [".png", ".jpg", ".jpeg"]
    for a in accepted:
      if fname.endswith(a):
        return True
    return False

  def _sample_one(self, iA):
    # iA = np.random.randint(0, self._len_A)
    img_A = Image.open(self.domain_A[iA])
    img_A = img_A.convert('RGB') if img_A.mode == "RGBA" else img_A
    img_A = np.asarray(img_A)
    
    img_A = np.asarray(cv2.resize(img_A, (self.W, self.H))) if (self.W, self.H) != img_A.shape[:2] else img_A
    return {"image": img_A}

  def _sample(self, idx, current):
    batch = []
    for i in range(self._batch_size):
      data = self._sample_one(idx[(current + i) % len(self.domain_A)])
      batch.append(data)
    data = {k: np.stack([batch[i][k] for i in range(self._batch_size)], 0) for k in batch[0].keys()}
    t = np.random.randint(0, self._steps, (self._batch_size,))
    return {**data, **self.beta_sampler(t), "class": np.zeros((self._batch_size,))}

  def dataset(self):
    idx = np.random.permutation(np.arange(0, len(self.domain_A)))
    current = 0
    while True:
      out = self._sample(idx, current)
      current += self._batch_size
      if current >= len(self.domain_A):
        current = 0
        idx = np.random.permutation(np.arange(0, len(self.domain_A)))
        self.epoch_cnt += 1
      yield out