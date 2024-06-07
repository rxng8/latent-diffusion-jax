
import pathlib
import embodied


def load_config(argv: list) -> embodied.Config:
  config = embodied.api.load_config(pathlib.Path(__file__).parent / "configs.yaml", argv)
  print(config, '\n')
  print(f"logdir: {embodied.Path(config.logdir)}")
  return config

