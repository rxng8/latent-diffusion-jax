# """
# File: train.py
# Author: Viet Nguyen
# Date: 2024-06-01

# Description: This is the top-most file to conduct the training logic for CycleGAN model
# """

# # %%

# import importlib
# import pathlib
# import sys
# import warnings
# from functools import partial as bind
# import os

# warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
# warnings.filterwarnings('ignore', '.*using stateful random seeds*')
# warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
# warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
# warnings.filterwarnings('ignore', '.*RGB-array rendering should return a numpy array.*')
# warnings.filterwarnings('ignore', '.*Conversion of an array with ndim > 0 to a scalar is deprecated*')

# directory = pathlib.Path(__file__).resolve()
# directory = directory.parent
# sys.path.append(str(directory.parent))

# import embodied

# from cyclegan import train_eval, make_trainer, make_dataloader

# def main(argv):
#   config = embodied.api.load_config(pathlib.Path(__file__).parent / "configs.yaml", argv)
#   print(config, '\n')
#   print(f"logdir: {embodied.Path(config.logdir)}")

#   if config.run.script == "train":
#     train_eval(
#       make_trainer=bind(make_trainer, config),
#       make_dataloader_train=bind(make_dataloader, config),
#       make_dataloader_eval=bind(make_dataloader, config),
#       make_logger=bind(embodied.api.make_logger, config),
#       config=config
#     )
#   else:
#     raise NotImplementedError("")


# if __name__ == '__main__':
#   if embodied.check_vscode_interactive():
#     _args = [
#       "--expname=test",
#       "--configs=m2p,tiny",
#       "--run.steps=2000",
#       # "--run.from_checkpoint=logs/test/checkpoint.ckpt"
#     ]
#     main(_args)
#   else:
#     main(sys.argv[1:])

