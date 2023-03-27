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
        return Readout(2)(x)  # 1. Replace output layer with Readout layer


mup = Mup()

init_input = jnp.zeros(123)
base_model = MyModel(width=1)
base_variables = base_model.init(jax.random.PRNGKey(0), init_input)
mup.set_base_variables(base_variables)

model = MyModel(width=100)
params = model.init(jax.random.PRNGKey(0), init_input)
params = mup.set_target_variables(params)

optimizer = adam(3e-4)
optimizer = mup.wrap_optimizer(optimizer, adam=True)  # 6. Use wrap_optimizer to get layer specific learning rates

# Now the model can be trained as normal
