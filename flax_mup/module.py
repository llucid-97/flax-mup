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


class ReadoutAffine(nn.Module):
    """Wrapper around Affine. Used by Mup to set different learning rate."""
    feature_axes: nn.normalization.Axes = -1
    kernel_init: T.Callable = nn.initializers.lecun_normal()
    bias_init: T.Callable = nn.initializers.zeros
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        inputs /= self.variable('mup', 'divisor', jnp.ones, tuple()).value

        result = Affine(self.feature_axes,self.kernel_init,self.bias_init,self.use_bias)(inputs)
        return result

class Affine(nn.Module):
    """Mup Affine Transformation. Used by Mup to set different learning rate."""
    feature_axes: nn.normalization.Axes = -1
    kernel_init: T.Callable = nn.initializers.lecun_normal()
    bias_init: T.Callable = nn.initializers.zeros
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        axes = nn.normalization._canonicalize_axes(x.ndim, self.feature_axes)
        shape = [1] * x.ndim
        reduced_shape = []
        for ax in axes:
            shape[ax] = x.shape[ax]
            reduced_shape.append(x.shape[ax])

        kernel = self.param('kernel',self.kernel_init,shape,x.dtype)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, shape, x.dtype)
            return x * kernel + bias
        else:
            return x * kernel