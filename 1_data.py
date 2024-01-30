import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4' # Use 4 CPU devices

from utils import *
import jax
import jax.numpy as jnp
from tqdm import tqdm

def train_with_data_parallel(dataset: jnp.ndarray, params: list[Params], num_epochs: int):
    G = jax.local_device_count()
    # replicate model weights
    replicated_params = [
        jax.tree_map(lambda p: jnp.tile(p, (G, 1, 1)), param) 
        for param in params
    ]
    for epoch in range(num_epochs):
        avg_loss = 0
        for (x, y) in tqdm(dataset, leave=False):
            # shard data batch
            x, y = split(x, G), split(y, G)
            replicated_params, loss = update(replicated_params, x, y)
            # note that loss is actually an array of shape [G], with identical
            # entries, because each device returns its copy of the loss
            # visualize(loss) will show [CPU 0, CPU 1, ..., CPU G]
            avg_loss += loss.mean().item()
        if (epoch + 1) % 5 == 0:
            print(f"Step {epoch + 1:3d}, loss: {avg_loss / dataset.shape[0]:.3f}")
    return replicated_params

if __name__ == '__main__':
    num_epochs = 50
    num_samples = 500
    B, d, h, L = 20, 2, 4, 16
    # create random dataset
    dataset = create_dataset(num_samples, B, d)
    print('Dataset size:', dataset.shape) # [N, 2, B, d]
    # initialize weights
    params = init_weights(d, h, L, random.PRNGKey(42))
    # run training with DP
    train_with_data_parallel(dataset, params, num_epochs)