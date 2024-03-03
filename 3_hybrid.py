import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4' # Use 4 CPU devices

from utils import *
import jax
import jax.numpy as jnp
from tqdm import tqdm

def train_with_hybrid_parallel(dataset: ArrayLike, params: list[Params], num_epochs: int, DP: int, TP: int):
    sharded_params = [
        Params(w1=split(p.w1, num_sections=TP, axis=1), 
               w2=split(p.w2, num_sections=TP, axis=0))
        for p in params
    ]
    hybrid_params = [
        jax.tree_map(lambda param: jnp.tile(param, (DP, 1, 1)), p) 
        for p in sharded_params
    ]
    for epoch in range(num_epochs):
        avg_loss = 0
        for (x, y) in tqdm(dataset, leave=False):
            # shard and then replicate data batch
            x, y = split(x, DP), split(y, DP)
            x, y = jnp.repeat(x, TP, axis=0), jnp.repeat(y, TP, axis=0)
            hybrid_params, loss = update(hybrid_params, x, y)
            avg_loss += loss.mean().item()
        if (epoch + 1) % 5 == 0:
            print(f"Step {epoch + 1:3d}, loss: {avg_loss / dataset.shape[0]:.3f}")
    return hybrid_params

if __name__ == '__main__':
    num_epochs = 50
    num_samples = 500
    B, d, h, L = 20, 2, 4, 16
    G = jax.local_device_count()
    TP = 2
    DP = G // TP
    # create random dataset
    dataset = create_dataset(num_samples, B, d)
    print('Dataset size:', dataset.shape) # [N, 2, B, d]
    # initialize weights
    params = init_weights(d, h, L, random.PRNGKey(42))
    # run training with DP and TP
    train_with_hybrid_parallel(dataset, params, num_epochs, DP, TP)
