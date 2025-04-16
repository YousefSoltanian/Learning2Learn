import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

class StaticIntentModule(nn.Module):
    """A differentiable module for learning static intent parameters.

    The module learns a distribution over a static parameter vector. The distribution is 
    Gaussian with learnable mean and variance. By default (output_dim=1), it outputs a scalar,
    but you can specify a higher dimension to learn a vector of parameters.

    Attributes:
        output_dim (int): Dimensionality of the static parameter vector. (Default: 1)
    """
    output_dim: int = 1

    @nn.compact
    def __call__(self, rng, sample: bool = True):
        # Learnable parameters: mean and log variance (for stability).
        mean = self.param("mean", nn.initializers.normal(stddev=0.1), (self.output_dim,))
        log_var = self.param("log_var", nn.initializers.zeros, (self.output_dim,))
        
        # Ensure variance is positive.
        var = nn.softplus(log_var) + 1e-6
        
        if sample:
            # Sample from a standard normal distribution and use reparameterization.
            eps = jax.random.normal(rng, mean.shape)
            sample_value = mean + jnp.sqrt(var) * eps
            return jnp.squeeze(sample_value), mean, var
        else:
            return mean, mean, var

# --- Example usage ---
if __name__ == "__main__":
    # Instantiate the module with default output_dim=1 (scalar) or override (e.g., output_dim=3)
    model = StaticIntentModule(output_dim=1)
    
    # Create a JAX PRNG key.
    rng = jax.random.PRNGKey(42)
    rng, init_rng, sample_rng = jax.random.split(rng, 3)
    
    # Initialize parameters.
    variables = model.init({'params': init_rng, 'sample': sample_rng}, sample_rng, sample=True)
    
    # Forward pass: sample the static parameter, as well as get the mean and variance.
    sample_value, mean, var = model.apply(variables, sample_rng, sample=True)
    print("Sample output:", sample_value)
    print("Mean output:", mean)
    print("Variance output:", var)
