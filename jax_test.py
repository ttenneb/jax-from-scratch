import os
# os.add_dll_directory(
#     "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin")

import numpy as np
import time
from jax.lib import xla_bridge
from jax import random, jit, device_put
import jax.numpy as jnp
import jax


key = random.PRNGKey(0)
size = 12000

x = random.normal(key, (size, size), dtype=jnp.float32)
print(type(x))
x = device_put(x)
start = time.time()
jnp.dot(x, x.T).block_until_ready
jnp.dot(x, x.T).block_until_ready
jnp.dot(x, x.T).block_until_ready
jnp.dot(x, x.T).block_until_ready
print(time.time() - start)

x = np.array(x)
start = time.time()
np.dot(x, x.T)
np.dot(x, x.T)
np.dot(x, x.T)
np.dot(x, x.T)
print(time.time() - start)
