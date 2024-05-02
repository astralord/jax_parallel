# When running on CPU you can always emulate an arbitrary number of devices with a nifty 
# --xla_force_host_platform_device_count XLA flag, e.g. by executing the following before importing JAX:
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
# This is especially useful for debugging and testing locally or even for prototyping in Colab 
# since a CPU runtime is faster to (re-)start.

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PositionalSharding
from utils import *

if __name__ == '__main__':
    # [CpuDevice(id=0), CpuDevice(id=1), CpuDevice(id=2), CpuDevice(id=3),
    #  CpuDevice(id=4), CpuDevice(id=5), CpuDevice(id=6), CpuDevice(id=7)]
    print(jax.devices())

    batch_size, embed_dim = 16, 8
    x = jnp.zeros((batch_size, embed_dim))
    print(x.devices()) # will output default device, e.g. {CpuDevice(id=0)}
    print(jax.device_put(x, jax.devices()[1]).devices()) # {CpuDevice(id=1)}
    
    sharding = PositionalSharding(jax.devices())
    G = jax.local_device_count()
    sharded_x = jax.device_put(x, sharding.reshape(1, G))
    visualize(sharded_x)

    replicated_x = jax.device_put(x, sharding.replicate(0))
    visualize(replicated_x, color_map="Pastel2_r")

    combined_x = jax.device_put(x, sharding.reshape(2, G // 2).replicate(0))
    visualize(combined_x)

    # set up toy example hyper-parameters
    B, d, h = 16, 8, 32
    # create random keys
    data_key = random.PRNGKey(0)
    weight_key = random.PRNGKey(42)

    x, y = sample_data(B, d, data_key)
    params = init_ffn_weights(d, h, weight_key)
    visualize(ffn(x, params))
    
    sharded_x = jax.device_put(x, sharding.reshape(G, 1))
    visualize(ffn(sharded_x, params))