# """
# File: data.py
# Author: Viet Nguyen
# Date: 2024-06-01

# Description: Data handling
# """

# import sys, pathlib
# import numpy as np
# import jax
# import jax.numpy as jnp
# import os
# from PIL import Image
# import cv2

# import embodied

# class ModalityDataset:
#   pass

# class OneDomainDataset:
#   def __init__(self, path_A: str|pathlib.Path|embodied.Path, image_size=(256, 256), batch_size=16) -> None:
#     self.path_A = pathlib.Path(path_A).resolve()
#     self.path_B = pathlib.Path(path_B).resolve()
#     print(f"loading 2 domains from {self.path_A} and {self.path_B}")
#     self.H = image_size[1]
#     self.W = image_size[0]
#     self._batch_size = batch_size

#     # get all image path
#     self.domain_A = []
#     for fname in os.listdir(self.path_A):
#       if self._check_image(fname):
#         self.domain_A.append(self.path_A / fname)
#     self._len_A = len(self.domain_A)

#     self.domain_B = []
#     for fname in os.listdir(self.path_B):
#       if self._check_image(fname):
#         self.domain_B.append(self.path_B / fname)
#     self._len_B = len(self.domain_B)

#   def _check_image(self, fname: str):
#     accepted = [".png", ".jpg", ".jpeg"]
#     for a in accepted:
#       if fname.endswith(a):
#         return True
#     return False

#   def _sample_one(self):
#     iA = np.random.randint(0, self._len_A)
#     iB = np.random.randint(0, self._len_B)
#     img_A = np.asarray(Image.open(self.domain_A[iA]))
#     img_A = np.asarray(cv2.resize(img_A, (self.W, self.H))) if (self.W, self.H) == img_A.shape[:2] else img_A
#     img_B = np.asarray(Image.open(self.domain_B[iB]))
#     img_B = np.asarray(cv2.resize(img_B, (self.W, self.H))) if (self.W, self.H) == img_B.shape[:2] else img_B
#     return {"A": img_A, "B": img_B}

#   def _sample(self):
#     batch = []
#     for _ in range(self._batch_size):
#       data = self._sample_one()
#       batch.append(data)
#     return {k: np.stack([batch[i][k] for i in range(self._batch_size)], 0) for k in batch[0].keys()}

#   def dataset(self):
#     while True:
#       yield self._sample()