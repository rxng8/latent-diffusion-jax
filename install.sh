#!/bin/bash -l


# Install required libraries
# This already include jax, tf, tfprob
pip install git+https://github.com/rxng8/embodied.git@main --force-reinstall

pip install Pillow opencv-python==4.9.0.80 matplotlib==3.9.0 opendatasets==0.1.22