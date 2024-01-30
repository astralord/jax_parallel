import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4' # Use 4 CPU devices

import jax
import jax.numpy as jnp
from utils import *
from jax import random

def pipeline_inference(params: list[Params], x: jnp.ndarray, M: int):
    '''
        Split input batch to M micro-batches and run PP forward pass
    '''
    B = x.shape[0]
    micro_batch_size = B // M
    # re-organize weights
    params = stack_stage_weights(params)
    # split input data to micro-batches
    x = split(x, M)
    # create shifting buffer
    state = jnp.zeros((G, micro_batch_size, d))
    y_pred = []
    for i in range(M + G - 1):
        from_prev_stage = jnp.concatenate([jnp.expand_dims(x[i], 0), state[:-1]])
        state = jax.pmap(model)(from_prev_stage, params)
        if i >= G - 1: # first micro-batch has passed through the last stage
            y_pred.append(state[-1])
    return jnp.array(y_pred).reshape(B, d)

if __name__ == '__main__':
    # setup hyperparameters
    num_samples = 1
    L = 16
    B, d, h = 20, 2, 4
    G = jax.local_device_count()
    M = 10 # number of micro-batches
    params = init_weights(d, h, L, random.PRNGKey(42))
    x, _ = sample_data(B, d, random.PRNGKey(0))
    y_pred = pipeline_inference(params, x, M)