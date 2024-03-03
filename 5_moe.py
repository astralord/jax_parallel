import jax
import jax.numpy as jnp
from jax import random
from utils import *

def dense_gating(x: ArrayLike, gate_params: ArrayLike) -> Array:
    return jax.nn.softmax(x @ gate_params)

def sparse_gating(x: ArrayLike, 
                  gate_params: ArrayLike,  
                  topk: int, 
                  noise_weights: ArrayLike=None,
                  rng: ArrayLike=None):
    h = x @ gate_params 
    if noise_weights is not None:
        assert rng is not None, "Random seed is required to use noisy gating"
        eps = random.normal(rng, h.shape)
        noise = eps * jax.nn.softplus(x @ noise_weights)
        h += noise
    _, top_k_ids = jax.lax.top_k(h, topk)
    mask = index_to_mask(top_k_ids, h.shape)
    h = jnp.where(mask, h, -jnp.inf)
    return jax.nn.softmax(h)

if __name__ == '__main__':
    B, d, E = 5, 2, 4
    random_keys = random.split(random.PRNGKey(42), 4)
    x = random.normal(random_keys[0], (B, d))
    w_g = random.normal(random_keys[1], (d, E))

    print(f'Soft-Gating:\n {dense_gating(x, w_g)}\n')

    w_noise = random.normal(random_keys[2], (d, E))
    gates = sparse_gating(x, w_g, 2, w_noise, random_keys[3])
    print(f'Top-2 Sparse-Gating:\n{gates}\n')

    importance = jnp.sum(gates, axis=0)
    print(f'Importance: {importance}\n')
    cv = jnp.std(importance) / jnp.mean(importance)
    print(f'Load-balance loss: {cv ** 2}')