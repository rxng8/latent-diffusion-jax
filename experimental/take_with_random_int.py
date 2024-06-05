# %%

import jax
import jax.numpy as jnp
from jax import random, jit

key = random.PRNGKey(0)

# Define the function
@jit
def take_random_element():
    # Step 1: Set up a PRNG key with a seed
    
    
    # Step 2: Generate a random integer between 0 and array_size - 1
    random_index = random.randint(key, shape=(3,), minval=0, maxval=array_size)
    
    # Step 3: Create an array of given size
    array = jnp.arange(10, dtype=jnp.float32)
    
    # Step 4: Use jnp.take with the generated random integer
    result = jnp.take(array, random_index)
    
    return random_index, result

# Parameters
seed = 123
array_size = 10

# Call the JIT-compiled function
random_index, result = take_random_element()

print("Random index:", random_index)
print("Element at random index:", result)