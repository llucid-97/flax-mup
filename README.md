# MUP for Flax

This is a fork of Davis Yoshida's [haiku implementation](https://github.com/davisyoshida/haiku-mup) of Yang and Hu et al.'s [μP project](https://github.com/microsoft/mup), porting it to Flax.
It's not feature complete, and we're very open to suggestions on improving the usability.

**NOTE**: We have not yet added support for shared embedding layers.

## Installation

```
pip install git+https://github.com/llucid-97/flax-mup
```

## Learning rate demo
These plots show the evolution of the optimal learning rate for a 3-hidden-layer MLP on MNIST, trained for 10 epochs (5 trials per lr/width combination).

With standard parameterization, the learning rate optimum (w.r.t. training loss) continues changing as the width increases, but μP keeps it approximately fixed:

<img src="https://github.com/davisyoshida/haiku-mup/blob/master/figures/combined.png?raw=True" width="1024" />

Here's the same kind of plot for 3 layer transformers on the Penn Treebank, this time showing Validation loss instead of training loss, scaling both the number of heads and the embedding dimension simultaneously:

<img src="https://github.com/davisyoshida/haiku-mup/blob/master/figures/ptb_combined.png?raw=True" width="1024" />

Note that the optima have the same value for n_embd=80. That's because the other hyperparameters were tuned using an SP model with that width, so this shouldn't be biased in favor of μP.

## Usage

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from optax import adam

from flax_mup import Mup, Readout


class MyModel(nn.Module):
    width: int
    n_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.width)(x)
        x = jax.nn.relu(x)
        return Readout(2)(x)  # Replace output layer with Readout layer


mup = Mup()

init_input = jnp.zeros(123)
base_model = MyModel(width=1)
base_variables = base_model.init(jax.random.PRNGKey(0), init_input)
mup.set_base_shapes(base_variables)

model = MyModel(width=100)
params = model.init(jax.random.PRNGKey(0), init_input)
params = mup.set_target_shapes(params)

optimizer = adam(3e-4)
optimizer = mup.wrap_optimizer(optimizer, adam=True)  # Use wrap_optimizer to get layer specific learning rates

# Now the model can be trained as normal

```
