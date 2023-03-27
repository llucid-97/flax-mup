import jax.numpy as jnp
from flax import linen as nn
import typing as T


class Readout(nn.Module):
    """Wrapper around nn.Dense. Used by Mup to set different learning rate."""
    features: int
    use_bias: bool = True
    kernel_init: T.Callable = nn.initializers.lecun_normal()
    bias_init: T.Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        inputs /= self.variable('mup', 'divisor', jnp.ones, tuple()).value
        result = nn.Dense(features=self.features, kernel_init=self.kernel_init, bias_init=self.bias_init,
                          use_bias=self.use_bias)(inputs)

        return result