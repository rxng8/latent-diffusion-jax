"""
File: train.py
Author: Viet Nguyen
Date: 2024-06-01

Description: This is the top-most file to conduct the training logic for CycleGAN model
"""

# %%

import importlib
import pathlib
import sys
import warnings
from functools import partial as bind
import numpy as np
import os
import cv2
from PIL import Image

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
warnings.filterwarnings('ignore', '.*RGB-array rendering should return a numpy array.*')
warnings.filterwarnings('ignore', '.*Conversion of an array with ndim > 0 to a scalar is deprecated*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

import embodied

from ldm import train_eval, make_trainer, Dataloader, load_config

class CelebaHQLoader(Dataloader):
  def __init__(self, config, mode="train") -> None:
    super().__init__(config, mode)
    
    dirpath = pathlib.Path(self.config.dir_path)
    assert self._mode in ("train", "val")
    setname = self._mode
    self.path_A = dirpath / setname / "female"
    self.path_B = dirpath / setname / "male"
    # female folder: 0
    self.domain_A = []
    for fname in os.listdir(self.path_A):
      if self._check_image(fname):
        self.domain_A.append(self.path_A / fname)
    self._len_A = len(self.domain_A)
    # male folder: 1
    self.domain_B = []
    for fname in os.listdir(self.path_B):
      if self._check_image(fname):
        self.domain_B.append(self.path_B / fname)
    self._len_B = len(self.domain_B)

    self._current = 0
    self._total_size = self._len_A + self._len_B
    self._permutation = np.random.permutation(np.arange(0, self._total_size))

  def _check_image(self, fname: str):
    accepted = [".png", ".jpg", ".jpeg"]
    for a in accepted:
      if fname.endswith(a):
        return True
    return False
  
  def _load_image(self, imgpath: str) -> np.ndarray:
    img = np.asarray(Image.open(imgpath))
    img = np.asarray(cv2.resize(img, (self._width, self._height))) if (self._height, self._width) != img.shape[:2] else img
    return img # (uint8) [0, 255] # (H, W, C)

  def sample(self):
    batch = []
    for i in range(self._batch_size):
      id = self._permutation[(self._current + i) % (self._total_size)]
      if id < self._len_A:
        img = self._load_image(self.domain_A[id])
        imgcls = np.zeros(())
      else:
        img = self._load_image(self.domain_B[id - self._len_A])
        imgcls = np.ones(())
      batch.append({"image": img, "class": imgcls})
    # increase the current
    self._current += self._batch_size
    # reset permutation if finished the whole dataset
    if self._current >= self._total_size:
      self._permutation = np.random.permutation(np.arange(0, self._total_size))
    # finally, return the dictionary of batched data
    return {k: np.stack([batch[i][k] for i in range(self._batch_size)], 0) for k in batch[0].keys()}
    

def main(argv):
  config = load_config(argv)
  if config.run.script == "train":
    train_eval(
      make_trainer=bind(make_trainer, config),
      make_dataloader_train=lambda: CelebaHQLoader(config, mode="train"),
      make_dataloader_eval=lambda: CelebaHQLoader(config, mode="val"),
      make_logger=bind(embodied.api.make_logger, config),
      config=config
    )
  else:
    raise NotImplementedError("")


if __name__ == '__main__':
  if embodied.check_vscode_interactive():
    _args = [
      "--expname=test",
      "--configs=monet,tiny",
      "--run.steps=20000",
      # "--run.from_checkpoint=logs/test/checkpoint.ckpt"
    ]
    main(_args)
  else:
    main(sys.argv[1:])

