from PIL import Image
import numpy as np
import jax.numpy as jnp
transform = lambda x: x / 255.0 * 2 - 1
untransform = lambda x: ((x / 2 + 0.5) * 255.0).astype(np.uint8)
img = Image.open("aux/elefant.jpg")
img = img.resize((64, 64))
img = transform(np.asarray(img))
img = jnp.asarray(img)[None] # (1, 64, 64, 3) [-1, 1]