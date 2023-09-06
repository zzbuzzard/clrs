"""A cross example memory unit."""

import abc

import chex

import haiku as hk
import jax
import jax.numpy as jnp

_Array = chex.Array


class CrossExampleMemory(hk.Module):
  """Memory unit base class."""

  def __init__(self, name: str = 'memory'):
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

  @abc.abstractmethod
  def new_epoch(self):
    """
    Alerts the data structure to the beginning of a new epoch.
    """
    raise NotImplementedError


class MeanStdMemory(CrossExampleMemory):
  def __init__(self, size: int, dim: int):
    """
    Args:
      size: Number of entries to store (should equal epoch size)
      dim: Dimensionality of entries (should equal node feature dimensionality)
    """
    super().__init__("mean_std_memory")
    self._size = size
    self._dim = dim

  def transform_features(self, node_fts: _Array) -> _Array:
    means = hk.get_state("means", (self._size, self._dim), init=jnp.zeros())
    stds = hk.get_state("stds", (self._size, self._dim), init=jnp.zeros())

    temp = hk.get_parameter("temp1", shape=[], init=jnp.zeros())

    mean = jnp.mean(node_fts)
    std = jnp.std(node_fts)

    # Compute distance of node_fts to everything in the data structure
    ds = jnp.linalg.norm(means - mean, axis=1) + jnp.linalg.norm(stds - std, axis=1)

    # Compute strengths for each data structure item; a lower distance gives a higher strength
    s = jnp.exp(temp) / jnp.square(ds)
    w = jax.nn.softmax(s)

    # Goal mean/std are a linear combination of all items in data structure, weighted by w
    mean_goal = w * means
    std_goal = w * stds

    # Compute lerp factor; formulation 1
    temp2 = hk.get_parameter("temp2", shape=[], init=jnp.zeros())
    temp3 = hk.get_parameter("temp3", shape=[], init=jnp.zeros())
    lerp_factor = jax.nn.sigmoid(-jnp.exp(temp2) + jnp.exp(temp3) * jnp.sum(s))

    mean_final = lerp_factor * mean_goal + (1 - lerp_factor) * mean
    std_final = lerp_factor * std_goal + (1 - lerp_factor) * std

    node_fts_transformed = std_final * (node_fts - mean) / std + mean_final
    return node_fts_transformed

  def insert(self, node_fts: _Array):
    # Update new_means, new_stds and counter
    new_means = hk.get_state("new_means", (self._size, self._dim), init=jnp.zeros())
    new_stds = hk.get_state("new_stds", (self._size, self._dim), init=jnp.zeros())
    counter = hk.get_state("counter", [], dtype=jnp.int32, init=jnp.zeros())

    mean = jnp.mean(node_fts)
    std = jnp.std(node_fts)

    new_means = new_means.at[counter, :].set(mean)
    new_stds = new_stds.at[counter, :].set(std)

    hk.set_state("new_means", new_means)
    hk.set_state("new_stds", new_stds)
    hk.set_state("counter", counter + 1)

  def new_epoch(self):
    # Set means and stds to new_means and new_stds
    new_means = hk.get_state("new_means", (self._size, self._dim), init=jnp.zeros())
    new_stds = hk.get_state("new_stds", (self._size, self._dim), init=jnp.zeros())

    hk.set_state("means", new_means)
    hk.set_state("stds", new_stds)
    hk.set_state("counter", 0)
