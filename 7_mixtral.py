import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

from utils import *
from jax.sharding import PositionalSharding
moe = __import__('5_moe')

class SwiGLUParams(NamedTuple):
    w1: jnp.ndarray
    w2: jnp.ndarray
    v:  jnp.ndarray

@jit
def swiglu(x: ArrayLike, params: SwiGLUParams) -> Array:
    y = x @ params.v
    z = jax.nn.swish(x @ params.w1)
    return (z * y) @ params.w2

def init_swiglu_weights(embed_dim: int, hidden_dim: int, rng: ArrayLike) -> SwiGLUParams:
    '''
        Create SwiGLU weights with Xavier initialization
    '''
    std = jnp.sqrt(2 / (embed_dim + hidden_dim))
    w1_key, w2_key, v_key = random.split(rng, 3)
    w1 = std * random.normal(w1_key, (embed_dim, hidden_dim))
    w2 = std * random.normal(w2_key, (hidden_dim, embed_dim))
    v  = std * random.normal(v_key,  (embed_dim, hidden_dim))
    return SwiGLUParams(w1, w2, v)

if __name__ == '__main__':
    G = jax.local_device_count()
    sharding = PositionalSharding(jax.devices())

    # set up toy example hyper-parameters
    B, d, h, E = 16, 8, 32, 8
    # create random keys
    data_key = random.PRNGKey(0)
    weight_key = random.PRNGKey(42)

    x, _ = sample_data(B, d, data_key)
    sharded_x = jax.device_put(x, sharding.reshape(G, 1))
    params = init_swiglu_weights(d, h, weight_key)
    visualize(ffn(sharded_x, params))

    w_g = random.normal(weight_key, (d, E))
    moe.sparse_gating(x, w_g, topk=2)