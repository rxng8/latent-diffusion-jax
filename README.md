<p align="center">
  <img src="resources/figs/rickroll.gif" />
</p>


# Latent Diffusion Model

A simple and organized implementation of the Latent Diffusion Model algorithm


* How to run:
```bash
conda create -n ldm python=3.10
conda activate ldm
bash install.sh
python scripts/train.py
```

* To download the celeba-hq dataset:
  * Go to kaggle, create a key (in setting)
  * download the file, put in `.kaggle/kaggle.json`
  * Do some unzipping:

```bash
mkdir -p data && cd data
kaggle datasets download -d lamsimon/celebahq
unzip celebahq.zip -d celebahq
sudo rm -rf celebahq.zip
cd ..
```

* This pipeline is designed so that it is takes in a dataloader, and automatically train for you! For example:
```python
import sys
import numpy as np
import embodied
from ldm import Dataloader, load_config
class DummyLoader(Dataloader):
  def __init__(self, config, mode="train") -> None:
    super().__init__(config, mode)
  
  def sample(self):
    return {
      "image": np.zeros((self._batch_size, self._height, self._width, 3)),
      "class": np.zeros((self._batch_size,))
    }

config = load_config(sys.argv[1:])
dataloader = DummyLoader(config, mode="train")
```

* implement/override the `sample()` method to return a dictionary of data with key:
  * `image` mapping to a value of shape (batch_size, height, width, channel)
  * `class` mapping to a label/classification of that image.
* In short: `sample(): () -> dict: ('image': (B, H, W, C), 'class': (B,))`. Image is: `uint8` having range of `[0, 255]`

* All usable variable from the parent class `Dataloader`:
```python
self._batch_size # the batch size specified in the config file
self._height # the image height specified in the config: config.image_size[1]
self._width # the image width specified in the config: config.image_size[0]
self._mode = mode # str: the mode switcher of the dataset, it is set but no used anywhere in the parent class
self.config = config # the main config variable, can be called directly from the yaml config file by duck-typing
```

* For example, a CelebaHQ dataloader could be implemented as below:

```python
from ldm import Dataloader

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
```

* Then in the run, just call:
```python
import sys
import embodied
from ldm import train_eval, load_config, make_trainer

config = load_config(sys.argv[1:])
train_eval(
  make_trainer=bind(make_trainer, config),
  make_dataloader_train=lambda: CelebaHQLoader(config, mode="train"),
  make_dataloader_eval=lambda: CelebaHQLoader(config, mode="val"),
  make_logger=bind(embodied.api.make_logger, config),
  config=config
)

```

* We can then run the train.py script using the dir_path flag to specify the dataset
```bash
python scripts/train.py --dir_path data/celebahq/celeba_hq --expname my_celeba_hq_experiment
```

* To see and monitor the result on tensorboard, run: `tensorboard --logdir logs --port 6006`, and then go to `http://localhost:6006` to the see result.