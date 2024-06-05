"""
File: data.py
Author: Viet Nguyen
Date: 2024-06-04

Description: Data handling
"""

import sys, pathlib
import numpy as np
import jax
import jax.numpy as jnp
import os
from PIL import Image
import cv2

import embodied


class ElefantDataset:
  def __init__(self, image_size=(64, 64), batch_size=16) -> None:
    self._batch_size = batch_size
    elefant = pathlib.Path(__file__).parent / "elefant.jpg"
    elefant = Image.open(elefant)
    elefant = elefant.resize(image_size)
    elefant = np.asarray(elefant)[None]
    elefant = np.repeat(elefant, batch_size, 0) # (B, H, W, C)
    self.elefant = {"image": elefant, "class": np.zeros((batch_size))}

  def _sample(self):
    return self.elefant.copy()

  def dataset(self):
    while True:
      yield self._sample()

class ModalityDataset:
  pass

class SingleDomainDataset:
  def __init__(self, dir_path: str|pathlib.Path|embodied.Path, image_size=(256, 256), batch_size=16) -> None:
    self.path_A = pathlib.Path(dir_path).resolve()
    print(f"loading 1 domains from {self.path_A}")
    self.H = image_size[1]
    self.W = image_size[0]
    self._batch_size = batch_size
    self.epoch_cnt = 0

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
    data["class"] = np.zeros((self._batch_size,))

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
  def __init__(self, path_A: str|pathlib.Path|embodied.Path, path_B: str|pathlib.Path|embodied.Path, image_size=(256, 256), batch_size=16) -> None:
    self.path_A = pathlib.Path(path_A).resolve()
    self.path_B = pathlib.Path(path_B).resolve()
    print(f"loading 1 domains from {self.path_A}")
    print(f"loading 1 domains from {self.path_B}")
    self.H = image_size[1]
    self.W = image_size[0]
    self._batch_size = batch_size
    self.epoch_cnt = 0

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
    return {k: np.stack([batch[i][k] for i in range(self._batch_size)], 0) for k in batch[0].keys()}

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