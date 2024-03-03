import jax
import jax.numpy as jnp
from jax import jit, random
from jax.debug import visualize_array_sharding
import matplotlib as mpl
from typing import NamedTuple
from jax._src.typing import ArrayLike, Array, Any
import functools

class Params(NamedTuple):
    w1: jnp.ndarray
    w2: jnp.ndarray

def visualize(tensor: ArrayLike, color_map: str="Set3"):
    visualize_array_sharding(tensor, color_map=mpl.colormaps[color_map])

@jit
def ffn(x: ArrayLike, params: Params) -> Array:
    z = jnp.maximum(x @ params.w1, 0)
    return z @ params.w2

@jit
def model(x: ArrayLike, params: Params) -> Array:
    for p in params:
        x += ffn(x, p)
    return x

def init_ffn_weights(embed_dim: int, hidden_dim: int, rng: ArrayLike) -> Params:
    '''
        Create FFN weights with Xavier initialization
    '''
    std = jnp.sqrt(2 / (embed_dim + hidden_dim))
    w1_key, w2_key = random.split(rng)
    w1 = std * random.normal(w1_key, (embed_dim, hidden_dim))
    w2 = std * random.normal(w2_key, (hidden_dim, embed_dim))
    return Params(w1, w2)

def init_weights(embed_dim: int, hidden_dim: int, layer_num: int, rng: ArrayLike) -> list[Params]:
    '''
        Create weights for a stack of `layer_num` FFN layers
    '''
    layer_keys = random.split(rng, layer_num)
    return [
        init_ffn_weights(embed_dim, hidden_dim, layer_keys[l]) 
        for l in range(layer_num)
    ]

def sample_data(batch_size: int, embed_dim: int, rng: ArrayLike) -> tuple[Array, Array]:
    '''
        Create random features `x` and dependable random targets `y`
    '''
    x = random.normal(rng, (batch_size, embed_dim))
    w = random.normal(random.PRNGKey(1), (embed_dim, embed_dim))
    y = jnp.sin(x @ w)
    return x, y

def create_dataset(num_samples: int, batch_size: int, embed_dim: int) -> Array:
    '''
        Create an array of samples (`x`, `y`)
    '''
    return jnp.array([
        sample_data(batch_size, embed_dim, random.PRNGKey(i)) 
        for i in range(num_samples)
    ])

@jit 
def criterion(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    return jnp.mean((y_pred - y_true) ** 2)

@jit
def loss_fn(params: Params, x: ArrayLike, y: ArrayLike) -> float:
    y_pred = model(x, params)
    return criterion(y_pred, y)

# Remember that the 'G' is just an arbitrary string label used
# to later tell 'jax.lax.pmean' which axis to reduce over. Here, we call it
# 'G', but could have used anything, so long as 'pmean' used the same.
@functools.partial(jax.pmap, axis_name='G')
def update(params: Params, x: ArrayLike, y: ArrayLike) -> Any:
    # Compute the gradients on the given minibatch (individually on each device)
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

    # Combine the gradient across all devices (by taking their mean)
    grads = jax.lax.pmean(grads, axis_name='G')

    # Also combine the loss. Unnecessary for the update, but useful for logging
    loss = jax.lax.pmean(loss, axis_name='G')

    # Each device performs its own update, but since we start with the same params
    # and synchronise gradients, the params stay in sync
    LEARNING_RATE = 1e-3
    new_params = jax.tree_map(
       lambda param, g: param - g * LEARNING_RATE, params, grads)
    return new_params, loss

def split(arr: ArrayLike, num_sections: int=None, axis: int=0) -> Array:
    return jnp.array(jnp.split(arr, num_sections, axis=axis))

def stack_stage_weights(params: list[Params]) -> list[Params]:
    '''
        Stack G stages, each containing L/G FFN layers
    '''
    L = len(params)
    G = jax.local_device_count()
    assert L % G == 0, f"Number of layers {L} must be divisible by number of stages {G}"
    stage_layers = L // G
    out_params = []
    for l in range(stage_layers):
        w1 = jnp.stack([params[l + g * stage_layers].w1 for g in range(G)])
        w2 = jnp.stack([params[l + g * stage_layers].w2 for g in range(G)])
        out_params.append(Params(w1, w2))
    return out_params

def scatter(input: ArrayLike, dim: int, index: ArrayLike, src: int) -> Array:
    '''
        Scatter function analogous to PyTorch `scatter_`
    '''
    idx = jnp.meshgrid(*(jnp.arange(n) for n in input.shape), 
                       sparse=True, 
                       indexing='ij')
    idx[dim] = index
    return input.at[tuple(idx)].set(src)

def index_to_mask(index: ArrayLike, input_shape: tuple) -> Array:
    '''
        Transform given indices to mask of input shape,
        where mask[index] = True and False otherwise
    '''
    zeros = jnp.zeros(input_shape, dtype=bool)
    return scatter(zeros, 1, index, True)

if __name__ == '__main__':
    import os
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
    d, h = 8, 32
    G = jax.local_device_count()
    x = jnp.arange(G * d).reshape(G, d) # dummy sample
    params = init_ffn_weights(d, h, random.PRNGKey(42))
    # replicate model weights
    params = jax.tree_map(lambda p: jnp.tile(p, (G, 1, 1)), params)
    visualize(jax.pmap(ffn, axis_name='G')(x, params))