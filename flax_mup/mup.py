import dataclasses
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from dataclasses import dataclass
from flax import linen as nn
import typing as T
import haiku as hk
import jax
from jax import numpy as jnp
import optax
from argparse import Namespace
from .module import Readout


def get_shapes(params):
    return jax.tree_map(lambda p: p.shape, params)


def maybe_unfreeze(x):
    try:
        x = x.unfreeze()
    except AttributeError:
        pass
    return x


@dataclasses.dataclass
class Mup:
    """Class which tracks infinite shapes, and applies per-parameter learning rates/multipliers"""

    readout_zero_init: bool = True

    def set_base_shapes(self, variables):
        self.base_shapes = jax.tree_util.tree_map(lambda x: Namespace(shape=x.shape), maybe_unfreeze(variables))

    def set_target_shapes(self, variables):
        variables = maybe_unfreeze(variables)
        from functools import partial

        shapes = jax.tree_util.tree_map(lambda x: Namespace(shape=x.shape), maybe_unfreeze(variables))
        f_sgd = partial(self._get_inf_ratios, optimizer='sgd')
        self._sgd_lrs = jax.tree_util.tree_map(f_sgd, self.base_shapes, shapes)['params']

        f_adam = partial(self._get_inf_ratios, optimizer='adam')
        self._adam_lrs = jax.tree_util.tree_map(f_adam, self.base_shapes, shapes)['params']

        f_width_mults = partial(self._get_inf_ratios, optimizer=None, )
        self._width_mults = jax.tree_util.tree_map(f_width_mults, self.base_shapes, variables)

        from flatdict import FlatDict
        fdp = FlatDict(variables)
        wm = FlatDict(self._width_mults)
        for k in dict(fdp):
            if ("Readout" in k) and ("kernel" in k):
                # TODO: Split into Dense and k
                k: str
                path = k.split(':')[:-1]
                path[0] = 'mup'
                path[-1] = 'divisor'
                k_mup_divisor = ':'.join(path)
                fdp[k_mup_divisor] *= wm[k]
                if self.readout_zero_init:
                    fdp[k] = fdp[k] * 0.
                else:
                    fdp[k] = fdp[k] / ((1 / wm[k]) ** 0.5)

        return fdp.as_dict()

    def wrap_optimizer(self, optimizer, adam=True):
        """Apply the per-parameter learning rates computed by `init_context` to an Optax optimizer."""
        if not self._adam_lrs:
            raise ValueError(
                'Attempted to wrap optimizer before initializing network. Did you forget to use init_base/init_target/apply_mup?')

        def init_fn(params):
            return optax.EmptyState()

        def update_fn(updates, state, params=None):
            updates = jax.tree_map(
                lambda update, scale: update * scale,
                updates,
                self._adam_lrs if adam else self._sgd_lrs
            )

            return updates, state

        return optax.chain(
            optimizer,
            optax.GradientTransformation(init_fn, update_fn)
        )

    def _get_inf_ratios(self, base: tuple, target: tuple,
                        optimizer: T.Literal['sgd', 'adam', None] = None,
                        ):
        n_inf = sum(a != b for a, b in zip(base.shape, target.shape))
        if n_inf > 2:
            raise ValueError(f'At most two infinite dimensions supported. Found {n_inf}')

        inf_ratios = [b / a for a, b in zip(base.shape, target.shape) if a != b]
        # return n_inf, inf_ratios
        width_mult = 1 if n_inf == 0 else inf_ratios[0]
        if optimizer is None:
            return width_mult
        if n_inf == 2:
            fanin_fanout_ratio = width_mult / inf_ratios[1]
            return (1 / fanin_fanout_ratio) if optimizer == 'sgd' else (1 / width_mult)
        elif n_inf == 1:
            return (width_mult) if optimizer == 'sgd' else 1.
        else:
            return 1.
