from typing import Literal, TypedDict

import jax
import jax.flatten_util
import jax.numpy as jnp
from brax.training.types import Params


class CompressionSpec(TypedDict):
    method: Literal["top", "random"]
    k: float


def compress(
    compression_spec: CompressionSpec, rng: jax.Array, params: jax.Array
) -> Params:
    if compression_spec["k"] == 1:
        return params
    k = int(compression_spec["k"] * len(params))
    if compression_spec["method"] == "top":
        _, ids = jax.lax.top_k(params**2, k)
    elif compression_spec["method"] == "random":
        ids = jax.random.choice(rng, params.shape[0], shape=(k,), replace=False)
    else:
        raise NotImplementedError("Compression method not implemented")
    values = params[ids]
    outs = jnp.zeros_like(params)
    outs = outs.at[ids].set(values)
    return outs
