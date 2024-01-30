# When running on CPU you can always emulate an arbitrary number of devices with a nifty 
# --xla_force_host_platform_device_count XLA flag, e.g. by executing the following before importing JAX:
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4' # Use 4 CPU devices
# This is especially useful for debugging and testing locally or even for prototyping in Colab 
# since a CPU runtime is faster to (re-)start.

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PositionalSharding
from utils import *

def load_balance_loss(gating_probs, expert_mask):
    '''
        Calculate load−balancing loss to ensure diverse expert routing.
    '''
    # gating probs is the probability assigned for each expert per token
    # gating probs shape: [G, S, E]
    # expert index contains the expert with the highest gating
    # probability in one−hot format
    # expert mask shape: [G, S, E]
    # For each core, get the fraction of tokens routed to each expert
    # density_1 shape: [G, E]
    density_1 = jnp.mean(expert_mask, axis=1)
    # For each core, get fraction of probability mass assigned to each expert
    # from the router across all tokens.
    # density_1_proxy shape: [G, E]
    density_1_proxy = jnp.mean(gating_probs, axis=1)
    # density_1 for a single device: vector of length E that sums to 1.
    # density_1_proxy for a single device: vector of length E that sums to 1.
    # Want both vectors to have uniform allocation (1/E) across all E elements.
    # The two vectors will be pushed towards uniform allocation when the dot product 
    # is minimized.
    loss = jnp.mean(density_1_proxy * density_1) * (density_1.shape[-1] ** 2)
    return loss

def switch_gating(x, gate_params, expert_capacity):
    '''
        Core layout is split across G for all tensors and operations.
        x: [G, S, d] - sharded tensor
        gate_params: [d, E] - gating weights
        expert_capacity: float - defines limit for capacity dimension
    '''
    
    # Probabilities for each token of what expert it should be sent to.
    # gating_probs shape: [G, S, E]
    gating_probs = jax.nn.softmax(x @ gate_params)

    # Get the top−1 expert for each token. 
    # expert_gate is the probability from the gating to top-1 expert
    # expert_index is what expert each token is going to be routed to
    # expert_gate shape: [G, S]
    # expert_index shape: [G, S]
    expert_gate, expert_index = jax.lax.top_k(gating_probs, 1)
    expert_gate, expert_index = expert_gate.squeeze(), expert_index.squeeze()

    # expert_mask shape: [G, S, E]
    expert_mask = jax.nn.one_hot(expert_index, num_classes=gating_probs.shape[2])
    
    # Compute load balancing loss.
    aux_loss = load_balance_loss(gating_probs, expert_mask)
    
    # Experts have a fixed capacity C, ensure we do not exceed it. 
    # Construct the batch indices, to each expert, with position in expert
    # make sure that not more that C examples can be routed to each expert.
    position_in_expert = jnp.cumsum(expert_mask, axis=1) * expert_mask
    
    # Keep only tokens that fit within expert capacity.
    expert_mask_trunc = expert_mask * jnp.less(position_in_expert, expert_capacity)
    expert_mask_flat = jnp.sum(expert_mask_trunc, axis=2)

    # Mask out the experts that have overflowed the expert capacity.
    expert_gate *= expert_mask_flat

    # combine_tensor used for combining expert outputs and scaling with gating probability.
    # combine_tensor shape: [G, S, E, C]
    expert_capacity_int = int(jnp.ceil(expert_capacity))
    combine_tensor = (expert_gate[..., None, None] *
                      expert_mask[..., None] *
                      jax.nn.one_hot(position_in_expert, num_classes=expert_capacity_int))
    combine_tensor = combine_tensor[..., 1:] # cut 0-dimension which is always 0s
    dispatch_mask = combine_tensor.astype(bool)
    return combine_tensor, dispatch_mask, aux_loss

def switch_layer(x, gate_params, ffn_params, capacity_factor):
    '''
        Distributed Switch FFN layer
    '''
    # device layout: [G, 1, 1] −> [G, 1, 1, 1], [G, 1, 1, 1]
    # dispatch_mask (boolean) shape: [G, S, E, C]
    # dispatch_mask is used for routing tokens to the correct expert
    # combine_tensor (float) shape: [G, S, E, C]
    # combine_tensor used for combining expert outputs and scaling 
    # with gating probability.
    expert_capacity = x.shape[1] * capacity_factor / gate_params.shape[1]
    combine_tensor, dispatch_mask, aux_loss = switch_gating(x, gate_params, expert_capacity)

    # Matmul with large boolean tensor to assign tokens to the correct expert.
    # device layout: [G, 1, 1], −> [1, G, 1, 1]
    # expert inputs shape: [E, G, C, d]
    expert_inputs = jnp.einsum("GSEC,GSd->EGCd", dispatch_mask, x)

    # All−to−All communication. Cores split across G and now we want to split
    # across E. This sends tokens, routed locally, to the correct expert now
    # split across different cores.
    # device layout: [1, G, 1, 1] −> [G, 1, 1, 1]
    sharding = PositionalSharding(jax.devices())
    expert_inputs = jax.device_put(expert_inputs, sharding.reshape(G, 1, 1, 1))

    # Standard FFN computation, where each expert has
    # its own unique set of parameters.
    # Total unique parameters created: E * (d * h * 2).
    # expert_outputs shape: [E, G, C, d]
    expert_outputs = ffn(expert_inputs, ffn_params)
    
    # All−to−All communication. Cores are currently split across the experts
    # dimension, which needs to be switched back to being split across num cores.
    # device layout: [G, 1, 1, 1] −> [1, G, 1, 1]
    expert_outputs = jax.device_put(expert_outputs, sharding.reshape(1, G, 1, 1))

    # Convert back to input shape and multiply outputs of experts by the gating probability.
    # expert_outputs shape: [E, G, C, d]
    # expert_outputs_combined shape: [G, S, d]
    # device layout: [1, G, 1, 1] −> [G, 1, 1]
    expert_outputs_combined = jnp.einsum("EGCd,GSEC->GSd", expert_outputs, combine_tensor)
    return expert_outputs_combined, aux_loss


if __name__ == '__main__':
    B, d, h = 20, 2, 4
    G = jax.local_device_count()
    E, S = G, B // G
    random_keys = random.split(random.PRNGKey(42), 4)

    # create sharded tensor like in DP
    x = random.normal(random_keys[0], (G, S, d))
    sharding = PositionalSharding(jax.devices())
    x = jax.device_put(x, sharding.reshape(G, 1, 1))

    # create layer weights
    gate_params = random.normal(random_keys[1], (d, E))

    ffn_params = init_weights(d, h, E, random_keys[2])
    ffn_params = stack_stage_weights(ffn_params)[0]
    # represent experts as a sharded stack of E FFN layers
    ffn_params = Params(w1=jax.device_put(ffn_params.w1, sharding.reshape(G, 1, 1)),
                        w2=jax.device_put(ffn_params.w2, sharding.reshape(G, 1, 1)))

    y, aux_loss = switch_layer(x, gate_params, ffn_params, 2.0)
    print('Load-balancing loss:', aux_loss)
    visualize(y.reshape(-1, d))