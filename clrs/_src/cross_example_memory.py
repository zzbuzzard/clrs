"""A cross example memory unit."""

import abc

import chex

import haiku as hk
import jax
import jax.numpy as jnp
from jax import vmap

_Array = chex.Array

MEMORY_TAG = "_clrs_memory"


class CrossExampleMemory(hk.Module):
  """Memory unit base class."""

  def __init__(self, name: str = 'memory'):
    if not name.endswith(MEMORY_TAG):
      name = name + MEMORY_TAG
    super().__init__(name=name)

  @abc.abstractmethod
  def transform_features(self, node_fts: _Array) -> _Array:
    """
    Applies some transformation to the array of node features following a message passing step.

    Args:
      node_fts: Node features.

    Returns:
      Modified node features.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def insert(self, node_fts: _Array):
    """
    Inserts the given node features into the data structure.

    Args:
      node_fts: Node features.
    """
    raise NotImplementedError


class MeanStdMemory(CrossExampleMemory):
  def __init__(self, size: int, dim: int, use_std: bool = True):
    """
    Args:
      size: Number of entries to store.
      dim: Dimensionality of entries (should equal node feature dimensionality).
    """
    super().__init__("mean_std_clrs_memory")
    self._size = size
    self._dim = dim
    self._use_std = use_std
    self._vmap_transform_features = vmap(self._transform_features)

  def transform_features(self, node_fts: _Array) -> _Array:
    return self._vmap_transform_features(node_fts)

  def _transform_features(self, node_fts: _Array) -> _Array:
    size = (self._size, self._dim)
    means = hk.get_state("means", size, init=jnp.zeros)
    if self._use_std:
      stds = hk.get_state("stds", size, init=jnp.zeros)

    temp = hk.get_parameter("temp1", shape=[], init=jnp.ones)

    mean = jnp.mean(node_fts, axis=0)
    if self._use_std:
      std = jnp.std(node_fts, axis=0)

    # Compute distance of node_fts to everything in the data structure
    if self._use_std:
      ds = jnp.linalg.norm(means - mean, axis=1) + jnp.linalg.norm(stds - std, axis=1)
    else:
      ds = jnp.linalg.norm(means - mean, axis=1)

    # We are interested only in the closest N datapoints
    N = 50
    ds, inds_closest = jax.lax.top_k(-ds, N)  # simple workaround as jax.lax.min_k does not exist
    ds = -ds

    # Compute strengths for each data structure item; a lower distance gives a higher strength
    # s = jnp.exp(temp) / jnp.square(ds)
    s = jnp.exp(temp * -ds)
    w = jax.nn.softmax(s)

    # Goal mean/std are a linear combination of top N items in data structure, weighted by w
    means_closest = means[inds_closest]
    mean_goal = jnp.sum(w[:, None] * means_closest, axis=0)
    if self._use_std:
      stds_closest = stds[inds_closest]
      std_goal = jnp.sum(w[:, None] * stds_closest, axis=0)

    # Compute lerp factor; formulation 1
    # temp2 = hk.get_parameter("temp2", shape=[], init=jnp.zeros)
    # temp3 = hk.get_parameter("temp3", shape=[], init=jnp.zeros)
    # lerp_factor = jax.nn.sigmoid(-jnp.exp(temp2) + jnp.exp(temp3) * jnp.mean(s))

    # sigmoid(-1.0986122886681098) = 0.25
    temp2 = hk.get_parameter("temp2", shape=(), init=hk.initializers.Constant(-1.0986122886681098))
    lerp_factor = jax.nn.sigmoid(temp2)

    mean_final = lerp_factor * mean_goal + (1 - lerp_factor) * mean
    if self._use_std:
      std_final = lerp_factor * std_goal + (1 - lerp_factor) * std

    if self._use_std:
      node_fts_transformed = std_final * (node_fts - mean) / std + mean_final
    else:
      node_fts_transformed = node_fts - mean + mean_final

    return node_fts_transformed

  def insert(self, node_fts: _Array):
    counter = hk.get_state("counter", [], dtype=jnp.int32, init=jnp.zeros)
    means = hk.get_state("means", (self._size, self._dim), init=jnp.zeros)
    batch_means = jnp.mean(node_fts, axis=1)

    n = node_fts.shape[0]
    indices = (jnp.arange(n) + counter) % self._size

    means = means.at[indices].set(batch_means)
    hk.set_state("means", means)

    if self._use_std:
      stds = hk.get_state("stds", (self._size, self._dim), init=jnp.zeros)
      batch_stds = jnp.std(node_fts, axis=1)
      stds = stds.at[indices].set(batch_stds)
      hk.set_state("stds", stds)

    hk.set_state("counter", (counter + n) % self._size)


class PerNodeMemory(CrossExampleMemory):
  def __init__(self, size: int, dim: int):
    """
    Args:
      size: Number of entries to store.
      dim: Dimensionality of entries (should equal node feature dimensionality).
    """
    super().__init__("per_node_clrs_memory")
    self._size = size
    self._dim = dim
    self._vmap_transform_features = hk.vmap(hk.vmap(self._transform_feature, split_rng=False), split_rng=False)

  def transform_features(self, node_fts: _Array) -> _Array:
    return self._vmap_transform_features(node_fts)

  def _transform_feature(self, node_ft: _Array) -> _Array:
    """Transforms a single node feature."""
    data = hk.get_state("data", (self._size, self._dim), init=jnp.zeros)
    temp = hk.get_parameter("temp1", shape=[], init=jnp.ones)
    ds = jnp.linalg.norm(data - node_ft, axis=1)
    # We are interested only in the closest N datapoints
    # N = 50
    # ds, inds_closest = jax.lax.top_k(-ds, N)  # simple workaround as jax.lax.min_k does not exist
    # ds = -ds

    # Compute strengths for each data structure item; a lower distance gives a higher strength
    # s = jnp.exp(temp) / jnp.square(ds)
    s = jnp.exp(temp * -ds)
    w = jax.nn.softmax(s)

    # Goal mean/std are a linear combination of top N items in data structure, weighted by w
    data_closest = data#[inds_closest]
    data_goal = jnp.sum(w[:, None] * data_closest, axis=0)

    # Compute lerp factor; alternative formulation
    # temp2 = hk.get_parameter("temp2", shape=[], init=jnp.zeros)
    # temp3 = hk.get_parameter("temp3", shape=[], init=jnp.zeros)
    # lerp_factor = jax.nn.sigmoid(-jnp.exp(temp2) + jnp.exp(temp3) * jnp.mean(s))

    # sigmoid(-1.0986122886681098) = 0.25
    temp2 = hk.get_parameter("temp2", shape=(), init=hk.initializers.Constant(-1.0986122886681098))
    lerp_factor = jax.nn.sigmoid(temp2)

    return lerp_factor * data_goal + (1 - lerp_factor) * node_ft

  def insert(self, node_fts: _Array):
    data = hk.get_state("data", (self._size, self._dim), init=jnp.zeros)
    counter = hk.get_state("counter", [], dtype=jnp.int32, init=jnp.zeros)

    n = node_fts.shape[0] * node_fts.shape[1]
    indices = (jnp.arange(n) + counter) % self._size
    data = data.at[indices].set(node_fts.reshape((n, self._dim)))

    hk.set_state("data", data)
    hk.set_state("counter", (counter + n) % self._size)
