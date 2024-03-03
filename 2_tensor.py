import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4' # Use 4 CPU devices

from utils import *
import jax
import jax.numpy as jnp
from tqdm import tqdm

def train_with_tensor_parallel(dataset: ArrayLike, params: list[Params], num_epochs: int):
    G = jax.local_device_count()
    sharded_params = [
        Params(w1=split(p.w1, num_sections=G, axis=1), 
               w2=split(p.w2, num_sections=G, axis=0)) 
        for p in params
    ]
    for epoch in range(num_epochs):
        avg_loss = 0
        for (x, y) in tqdm(dataset, leave=False):
            # replicate data batch
            x, y = jnp.array([x] * G), jnp.array([y] * G)
            sharded_params, loss = update(sharded_params, x, y)
            avg_loss += loss.mean().item()
        if (epoch + 1) % 5 == 0:
            print(f"Step {epoch + 1:3d}, loss: {avg_loss / dataset.shape[0]:.3f}")
    return sharded_params

if __name__ == '__main__':
    num_epochs = 50
    num_samples = 500
    B, d, h, L = 20, 2, 4, 16
    # create random dataset
    dataset = create_dataset(num_samples, B, d)
    print('Dataset size:', dataset.shape) # [N, 2, B, d]
    # initialize weights
    params = init_weights(d, h, L, random.PRNGKey(42))
    # run training with TP
    train_with_tensor_parallel(dataset, params, num_epochs)