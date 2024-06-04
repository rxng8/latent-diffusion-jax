# %%

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import warnings
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

import embodied
from embodied import nn
from embodied.nn import ninjax as nj

warnings.filterwarnings('ignore', '.*input data to the valid range.*')

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

transform = lambda x: x / 255.0 * 2 - 1
untransform = lambda x: ((x / 2 + 0.5) * 255.0).astype(np.uint8)


class NoiseEstimatorUNet(nj.Module):

  stage: int = 3
  head: int = 1
  group: int = 1
  thidden: int = 32
  act: str = "gelu"

  def __init__(self, hidden: int) -> None:
    self._hidden = hidden

  def __call__(self, imgnoi: jax.Array, timeid: jax.Array, condition: jax.Array=None) -> jax.Array:
    # imgnoi (B, H, W, C), -> timeid: (B,) -> condition (B, H, W, C) or (B, S, C) -> noise (B, H, W, C)
    """_summary_

    Args:
        imgnoi (jax.Array): (B, H, W, C)
        timeid (jax.Array): (B,)
        condition (jax.Array): (B, S, Z) or (B, H, W, Z)

    Returns:
        jax.Array: (B, Z1)
    """
    if condition is None:
      condition = imgnoi
    B, H, W, C = imgnoi.shape
    x = self.get("conv", nn.Conv2D, self._hidden, 1)(imgnoi)
    time_embed = self.get("time", nn.TimeEmbedding, self.thidden)(timeid)
    carries = []

    #### downsampling
    dim = self._hidden * 2
    for s in range(self.stage):
      # two res blocks + self attention + add together
      x = self.get(f"dr{s}0", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"dr{s}1", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"da{s}", nn.CrossAttentionBlock, dim, head=self.head, group=self.group)(x, condition)
      carries.append(x)
      # Downsampling
      if s != self.stage - 1:
        x = self.get(f"dd{s}", nn.Conv2D, dim, 3, stride=2)(x)
      # increase dimension
      dim *= 2

    #### bottleneck
    x = self.get(f"br0", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
    x = self.get(f"ba", nn.CrossAttentionBlock, dim, head=self.head, group=self.group)(x, condition)
    x = self.get(f"br1", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)

    #### upsampling
    # Upsampling phase
    dim //= 2
    for s in reversed(range(self.stage)):
      x = jnp.concatenate([carries.pop(), x], -1)
      # two res blocks + self attention + add together
      x = self.get(f"ur{s}0", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"ur{s}1", nn.ResidualTimeBlock, dim, act=self.act, group=self.group)(x, time_embed)
      x = self.get(f"ua{s}", nn.CrossAttentionBlock, dim, head=self.head, group=self.group)(x, condition)
      # Upsampling
      if s != 0:
        x = self.get(f"uu{s}", nn.Conv2D, dim, 3, stride=2, transp=True)(x)
      # decrease dimension
      dim //= 2

    # Final ResNet block and output convolutional layer
    x = self.get(f"fr", nn.ResidualTimeBlock, self._hidden, act=self.act, group=self.group)(x, time_embed)
    x = self.get(f"out", nn.Conv2D, C, 1, stride=1)(x)
    return x

def test_noise_estimator_unet():
  N = NoiseEstimatorUNet(8, stage=5, head=8, group=8, name="U")
  B, H, W, C = 2, 64, 64, 3
  _, S, Z = 2, 40, 16
  img = jnp.asarray(np.random.normal(0, 1, (B, H, W, C)))
  cond = jnp.asarray(np.random.normal(0, 1, (B, S, Z)))
  tid = jnp.asarray(np.random.randint(0, 200, (B,)))

  params = nj.init(N)({}, img, tid, cond, seed=0)
  fn = jax.jit(nj.pure(N))
  _, out = fn(params, img, tid, cond)
  assert out.shape == (B, H, W, C)


class Diffuser(nj.Module):

  hidden: int = 8
  stage: int = 3
  head: int = 1
  group: int = 1
  thidden: int = 32
  act: str = "gelu"

  # implement the algorithm from https://arxiv.org/pdf/2006.11239.pdf
  # adapt from: https://github.com/andylolu2/jax-diffusion/blob/main/jax_diffusion/diffusion.py
  def __init__(self, beta_start: float, beta_final: float, steps: int):
    """_summary_

    Args:
      beta_start (float): the beta/variance of x_0 or the observation/frame/image
      beta_final (float): the beta/variance of x_T-1 or the latent/diffused noise
      steps (int): the total number of steps in the mdp
    """
    self._betas = np.linspace(beta_start, beta_final, steps) # (T,)
    self._alphas = 1 - self._betas # (T,)
    self._alpha_bars = np.cumprod(self._alphas) # (T,)
    self._steps = steps # ()
    self.unet = NoiseEstimatorUNet(self.hidden, stage = self.stage, head = self.head,
      group = self.group, thidden = self.thidden, act = self.act, name="unet")

  def forward(self, x_0: jax.Array, t: jax.Array) -> jax.Array:
    # given the image, add noise to it. See algorithm 1 in https://arxiv.org/pdf/2006.11239.pdf
    # For forward, we can compute a random t
    B, H, W, C = x_0.shape
    (B,) = t.shape
    """x_t, eps = self.sample_q(x_0, t): Samples x_t given x_0 by the q(x_t|x_0) formula."""
    # x_0: (B, H, W, C)
    alpha_bar_t = self._alpha_bars.take(t.astype(jnp.int32)) # (B,)
    alpha_bar_t = alpha_bar_t[: None, None] # (B, 1, 1)
    eps = jax.random.normal(nj.seed(), shape=x_0.shape, dtype=x_0.dtype)
    x_t = jnp.sqrt(alpha_bar_t) * x_0 + jnp.sqrt(1 - alpha_bar_t) * eps
    """end of x_t, eps = self.sample_q(x_0, t)"""
    return x_t.clip(-1, 1), eps

  def reverse_step(self, x_t: jax.Array, t: jax.Array):
    """See algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""
    B, H, W, C = x_t.shape
    (B,) = t.shape
    alpha_t = jnp.take(self._alphas, t)
    alpha_bar_t = jnp.take(self._alpha_bars, t)
    sigma_t = jnp.sqrt(jnp.take(self._betas, t))
    z = (t > 0) * jax.random.normal(nj.seed(), shape=x_t.shape, dtype=x_t.dtype)
    eps = self.unet(x_t, t)
    x = (1.0 / jnp.sqrt(alpha_t)) * (
      x_t - ((1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t)) * eps
    ) + sigma_t * z
    x = x.clip(-1, 1)
    return x, x

  def reverse(self, x_T: jax.Array) -> jax.Array:
    B, H, W, C = x_T.shape
    # given the noise, reconstruct the image
    # For reverse, we have to reverse it one by one
    ts = jnp.arange(0, self._steps)[..., None] # (T, 1)
    ts = jnp.repeat(ts, B, axis=1) # (T, B)
    x_hat_0, xs = nj.scan(self.reverse_step, x_T, ts, reverse=True, unroll=1, axis=0)
    return x_hat_0, xs
  

class Trainer(nj.Module):
  def __init__(self, config):
    self.config = config
    self.diffuser = Diffuser(**config.diffuser, name="diff")
    self.opt = nn.Optimizer(**config.opt, name="opt")
    self.modules = [self.diffuser]

  def train(self, data):
    opt_metrics, (outs, loss_metrics) = self.opt(self.modules,
      self.loss, data, has_aux=True)
    opt_metrics.update(loss_metrics)
    return outs, opt_metrics

  def infer(self, data):
    return self.diffuser.reverse(data["image"])

  def loss(self, data):
    x_0 = data["image"]
    B, H, W, C = x_0.shape

    # Generate random timesteps indices
    timesteps = np.random.randint(0, self.diffuser._steps, (B,))
    timesteps = nn.cast(timesteps)

    # Generating the noise and noisy image for this batch
    # Add noise to x_0 until timestep
    noisy_image, noise = self.diffuser.forward(x_0, timesteps)

    # Forward noising: given a noisy image, predict the noise added to that image
    pred_noise = self.diffuser.unet(noisy_image, timesteps)

    # l1 loss
    # loss = ((pred_noise - noise)**2).mean([-3, -2, -1]).mean()
    loss = ((pred_noise - noise)**2).mean()

    # metrics
    outs = {"pred_noise": pred_noise, "noise": noise, "noisy_image": noisy_image}
    metrics = {"loss": loss}
    return loss, (outs, metrics)

config = embodied.Config(
  diffuser=dict(
    beta_start=0.0003, beta_final=0.04, stage=5,
    steps=200, thidden=64, hidden=8, head=4, group=4, act="silu"
  ),
  opt=dict(
    lr=1e-4 # changing to larger learning rate make it nan
  )
)
T = Trainer(config, name="T")
img = Image.open("auxiliary/elefant.jpg")
img = img.resize((64, 64))
img = transform(np.asarray(img))
img = jnp.asarray(img)[None]
data = {"image": img}

params = nj.init(T.train)({}, data, seed=np.random.randint(0, 2**16))

# Purifying
train = jax.jit(nj.pure(T.train))
infer = jax.jit(nj.pure(T.infer))

losses = []

# %%

for i in range(20000): # 20000
  params, (out, mets) = train(params, data, seed=np.random.randint(0, 2**16))
  loss = mets['opt_loss']
  losses.append(loss)
  if i % 100 == 0:
    print(f"[Step {i+1}] Loss: {loss}")

plt.plot(losses)
plt.show



# %%

_, (outs, metrics) = train(params, data, seed=np.random.randint(0, 2**16))
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
pred_noise = outs["pred_noise"][0]
noise = outs["noise"][0]
noisy_image = outs["noisy_image"][0]
# recovered_image = outs["recovered_image"][0]
axes[0, 0].imshow(untransform(img[0]))
axes[0, 0].axis('off')
axes[0, 0].set_title("actual image")
axes[0, 1].imshow(untransform(pred_noise))
axes[0, 1].axis('off')
axes[0, 1].set_title("predicted noise")
axes[1, 0].imshow(untransform(noisy_image))
axes[1, 0].axis('off')
axes[1, 0].set_title("noisy_image")
axes[1, 1].imshow(untransform(noise))
axes[1, 1].axis('off')
axes[1, 1].set_title("actual noise")
plt.tight_layout()
plt.show()

print(f"Erorr check: {(noise - pred_noise).mean()}")

# %%

# reverse step
_, (x_0, xs) = infer(params, {"image": jnp.asarray(np.random.normal(0, 1, (1, 64, 64, 3)))}, seed=np.random.randint(0, 2**16))
# _, (x_0, xs) = infer(params, data, seed=gseed())

fig = plt.figure()
for i, tid in enumerate([299, 189, 150, 140, 90, 50, 10, 0]):
  plt.subplot(1, 8, i + 1)
  plt.imshow(untransform(xs[tid][0].clip(-1, 1)))
  plt.axis('off')
  plt.title(f"t={tid}")
plt.tight_layout
plt.show()


# %%







# %%
