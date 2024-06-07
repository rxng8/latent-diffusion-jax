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
  * `mkdir -p data && cd data`
  * do `kaggle datasets download -d lamsimon/celebahq`


