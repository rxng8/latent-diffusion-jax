"""
File: run.py
Author: Viet Nguyen
Date: 2024-06-01

Description: Runing Lo(ma)gic
"""

import sys, pathlib
import threading
import numpy as np
import chex
import jax
import jax.numpy as jnp
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from ruamel import yaml

from functools import partial as bind

import embodied
from embodied.nn import ninjax as nj
from embodied import nn
from embodied.nn import sg

from .data import SingleDomainDataset
from .trainer import DiffusionTrainer

# def fetch_async(value):
#   with jax._src.config.explicit_device_get_scope():
#     [x.copy_to_host_async() for x in jax.tree_util.tree_leaves(value)]
#   return value

# def take_mets(mets):
#   mets = jax.tree.map(lambda x: x.__array__(), mets)
#   mets = {k: v[0] for k, v in mets.items()}
#   mets = jax.tree.map(
#     lambda x: np.float32(x) if x.dtype == jnp.bfloat16 else x, mets)
#   return mets

class ParamsHandler:
  def __init__(self, params, mirrored):
    self._params = params
    self._mirrored = mirrored
    self.lock = threading.Lock()

  @embodied.timer.section('model_save')
  def save(self):
    with self.lock:
      return jax.device_get(self._params)

  @embodied.timer.section('model_load')
  def load(self, state):
    with self.lock:
      chex.assert_trees_all_equal_shapes(self._params, state)
      jax.tree.map(lambda x: x.delete(), self._params)
      self._params = jax.device_put(state, self._mirrored)

def make_trainer(config) -> DiffusionTrainer:
  return DiffusionTrainer(config, name="diff")

def make_dataloader(config) -> SingleDomainDataset:
  dataloader = SingleDomainDataset(config.dir_path, config.image_size, config.batch_size)
  return dataloader

def setup_jax(jaxcfg):
  try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.set_visible_devices([], 'TPU')
  except Exception as e:
    print('Could not disable TensorFlow devices:', e)
  if not jaxcfg.prealloc:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  xla_flags = []
  if jaxcfg.logical_cpus:
    count = jaxcfg.logical_cpus
    xla_flags.append(f'--xla_force_host_platform_device_count={count}')
  if jaxcfg.nvidia_flags:
    xla_flags.append('--xla_gpu_enable_latency_hiding_scheduler=true')
    xla_flags.append('--xla_gpu_enable_async_all_gather=true')
    xla_flags.append('--xla_gpu_enable_async_reduce_scatter=true')
    xla_flags.append('--xla_gpu_enable_triton_gemm=false')
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    os.environ['NCCL_IB_SL'] = '1'
    os.environ['NCCL_NVLS_ENABLE'] = '0'
    os.environ['CUDA_MODULE_LOADING'] = 'EAGER'
  if xla_flags:
    os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
  jax.config.update('jax_platform_name', jaxcfg.platform)
  jax.config.update('jax_disable_jit', not jaxcfg.jit)
  if jaxcfg.transfer_guard:
    jax.config.update('jax_transfer_guard', 'disallow')
  if jaxcfg.platform == 'cpu':
    jax.config.update('jax_disable_most_optimizations', jaxcfg.debug)
  embodied.nn.COMPUTE_DTYPE = getattr(jnp, jaxcfg.compute_dtype)
  print(f"COMPUTE_DTYPE set to {embodied.nn.COMPUTE_DTYPE}")
  embodied.nn.PARAM_DTYPE = getattr(jnp, jaxcfg.param_dtype)
  print(f"PARAM_DTYPE set to {embodied.nn.PARAM_DTYPE}")


def train_eval(make_trainer: callable, make_dataloader_train: callable, make_dataloader_eval: callable, make_logger: callable, config: embodied.Config):
  print('Logdir', config.logdir)

  RNG = np.random.default_rng(config.seed)
  def next_seed(sharding):
    shape = [2 * x for x in sharding.mesh.devices.shape]
    seeds = RNG.integers(0, np.iinfo(np.uint32).max, shape, np.uint32)
    return jax.device_put(seeds, sharding)

  # Making our components
  trainer: DiffusionTrainer = make_trainer()
  dataloader_train: SingleDomainDataset = make_dataloader_train()
  dataloader_eval: SingleDomainDataset = make_dataloader_eval()
  logger = make_logger()

  # Keep track/recoder of statistics
  step: embodied.Counter = logger.step
  usage = embodied.Usage(**config.run.usage)
  trainstats = embodied.Agg()
  train_fps = embodied.FPS()
  should_log = embodied.when.Clock(config.run.log_every)
  should_save = embodied.when.Clock(config.run.save_every)
  should_eval = embodied.when.Clock(config.run.eval_every)

  # Setup jax engine, transfer guard, and sharding for parallel training
  setup_jax(config.jax)
  available = jax.devices(config.jax.platform)
  embodied.print(f'JAX devices ({jax.local_device_count()}):', available)
  if config.jax.assert_num_devices > 0:
    assert len(available) == config.jax.assert_num_devices, (
        available, len(available), config.jax.assert_num_devices)
  train_devices = [available[i] for i in config.jax.train_devices]
  train_mesh = jax.sharding.Mesh(train_devices, 'i')
  train_sharded = jax.sharding.NamedSharding(train_mesh, jax.sharding.PartitionSpec('i'))
  train_mirrored = jax.sharding.NamedSharding(train_mesh, jax.sharding.PartitionSpec())
  print('Train devices: ', ', '.join([str(x) for x in train_devices]))


  # setup dataset and transform function
  def transform(data):
    return jax.device_put(data, train_sharded)
  dataset_train = iter(embodied.Prefetch(dataloader_train.dataset, transform, amount=2))
  dataset_eval = iter(embodied.Prefetch(dataloader_eval.dataset, transform, amount=2))

  # setup model and parameters
  params = nj.init(trainer.train)({}, next(dataset_train), seed=next_seed(train_sharded))
  params = jax.device_put(params, train_mirrored)
  params_handler = ParamsHandler(params, train_mirrored)

  # Load or save checkpoint
  checkpoint = embodied.Checkpoint(pathlib.Path(config.logdir) / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.params_handler = params_handler
  if config.run.from_checkpoint:
    checkpoint.load(config.run.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  # setup transformation of model training
  train = jax.jit(nj.pure(trainer.train))
  report = jax.jit(nj.pure(trainer.report))

  # Actual training loop
  while step < config.run.steps:
    # load next batch
    with embodied.timer.section('dataset_next'):
      batch = next(dataset_train)
    
    # Train one step
    params_handler._params, (outs, mets) = train(params_handler._params, batch, seed=next_seed(train_sharded))
    # Record fps
    train_fps.step(config.batch_size)
    # aggregate metrics
    # trainstats.add(take_mets(fetch_async(mets)), prefix='train')
    trainstats.add(jax.device_get(mets), prefix='train')
    # increase step
    step.increment()

    # if eval is needed
    if should_eval(step):
      _, mets = report(params_handler._params, next(dataset_eval), seed=next_seed(train_sharded))
      logger.add(jax.device_get(mets), prefix='report')

    # log if needed
    if should_log(step):
      logger.add(trainstats.result())
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/train': train_fps.result()})
      logger.write()

    if should_save(step):
      checkpoint.save()

  # Save the last step
  checkpoint.save()

  # Finally, close logger
  logger.close()


def train_eval_then_do_a_CARTWHEEL(make_trainer: callable, make_dataloader: callable, make_logger: callable, config: embodied.Config):
  raise NotImplementedError()

