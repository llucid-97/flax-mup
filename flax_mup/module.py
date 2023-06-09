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


class ReadoutConv(nn.Module):
    """Wrapper around nn.Dense. Used by Mup to set different learning rate."""
    features: int
    kernel_size: T.Sequence[int]
    strides: T.Union[None, int, T.Sequence[int]] = 1
    padding: nn.linear.PaddingLike = 'SAME'
    input_dilation: T.Union[None, int, T.Sequence[int]] = 1
    kernel_dilation: T.Union[None, int, T.Sequence[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    mask: T.Optional[nn.linear.Array] = None
    dtype: T.Optional[nn.linear.Dtype] = None
    param_dtype: nn.linear.Dtype = jnp.float32
    precision: nn.linear.PrecisionLike = None
    kernel_init: T.Callable[
        [nn.linear.PRNGKey, nn.linear.Shape, nn.linear.Dtype], nn.linear.Array] = nn.initializers.lecun_normal()
    bias_init: T.Callable[
        [nn.linear.PRNGKey, nn.linear.Shape, nn.linear.Dtype], nn.linear.Array] = nn.initializers.zeros
    module: nn.Module = nn.Conv

    @nn.compact
    def __call__(self, inputs):
        inputs /= self.variable('mup', 'divisor', jnp.ones, tuple()).value
        result = self.module(features=self.features,
                             kernel_size=self.kernel_size,
                             strides=self.strides,
                             padding=self.padding,
                             input_dilation=self.input_dilation,
                             kernel_dilation=self.kernel_dilation,
                             feature_group_count=self.feature_group_count,
                             use_bias=self.use_bias,
                             mask=self.mask,
                             dtype=self.dtype,
                             param_dtype=self.param_dtype,
                             precision=self.precision,
                             kernel_init=self.kernel_init,
                             bias_init=self.bias_init,
                             )(inputs)

        return result


class ReadoutAbstract(nn.Module):
    partial_module: T.Callable[[], nn.Module]

    @nn.compact
    def __call__(self, inputs):
        inputs /= self.variable('mup', 'divisor', jnp.ones, tuple()).value
        result = self.partial_module()(inputs)
        return result
