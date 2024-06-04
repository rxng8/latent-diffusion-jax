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
plt.tight_layout()
plt.show()


# %%







# %%
