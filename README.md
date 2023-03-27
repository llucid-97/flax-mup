# MUP for Flax

This is a fork of Davis Yoshida's [haiku implementation](https://github.com/davisyoshida/haiku-mup) of Yang and Hu et al.'s [μP project](https://github.com/microsoft/mup), porting it to Flax.
It's not feature complete, and we're very open to suggestions on improving the usability.

## Installation

```
pip install flax-mup
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
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from optax import adam, chain

from flax_mup import apply_mup, Mup, Readout

class MyModel(nn.Module):
    width: int
    n_classes: int = 10

    def __call__(self, x):
        x = nn.Dense(self.width)(x)
        x = jax.nn.relu(x)
        return Readout(2)(x) # 1. Replace output layer with Readout layer

def fn(x, width=100):
    with apply_mup(): # 2. Modify parameter creation with apply_mup()
        return MyModel(width=width)(x)

mup = Mup()

init_input = jnp.zeros(123)
base_model = MyModel(width=1)

model = MyModel(width=100)


model = mup.wrap_model(model) # 5. Modify your model with Mup

optimizer = adam(3e-4)
optimizer = mup.wrap_optimizer(optimizer, adam=True) # 6. Use wrap_optimizer to get layer specific learning rates

# Now the model can be trained as normal
```
### Summary
1. Replace output layers with `Readout` layers
2. Modify parameter creation with the `apply_mup()` context manager
3. Initialize a base model inside a `Mup.init_base()` context
4. Initialize the target model inside a `Mup.init_target()` context
5. Wrap the model with `Mup.wrap_model`
6. Wrap optimizer with `Mup.wrap_optimizer`

## Shared Input/Output embeddings
If you want to use the input embedding matrix as the output layer's weight matrix make the following two replacements:

```python
# old: embedding_layer = hk.Embed(*args, **kwargs)
# new:
embedding_layer = flax_mup.SharedEmbed(*args, **kwargs)
input_embeds = embedding_layer(x)

#old: output = hk.Linear(n_classes)(x)
# new:
output = flax_mup.SharedReadout()(embedding_layer.get_weights(), x) 
```
