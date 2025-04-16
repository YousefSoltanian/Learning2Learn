import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state
import optax  # for optimization
import numpy as np

class TransformerEncoderBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, train: bool):
        # Self-attention sub-layer:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not train
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = x + residual

        # Feed-forward sub-layer (MLP) with skip connection:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(4 * self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return x + residual


class BeliefUpdateTransformer(nn.Module):
    max_seq_len: int      # maximum sequence length (e.g. game horizon)
    state_dim: int        # dimension of each state observation (e.g. 6 for Lunar Lander)
    embed_dim: int        # embedding dimension
    transformer_layers: int
    num_heads: int
    mlp_hidden: int       # hidden size of the MLP head
    output_dim: int       # output belief parameters dimension (e.g. 2 for xf, yf)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, train: bool):
        batch_size, T, _ = x.shape

        # Embed each state observation.
        x = nn.Dense(self.embed_dim)(x)  # (batch, T, embed_dim)

        # Add learned positional embeddings.
        pos_embedding = self.param("pos_embedding",
                                   nn.initializers.normal(stddev=0.02),
                                   (self.max_seq_len, self.embed_dim))
        pos_emb = pos_embedding[:T, :]  # (T, embed_dim)
        pos_emb = jnp.expand_dims(pos_emb, axis=0)  # (1, T, embed_dim)
        x = x + pos_emb

        # Save a copy of the last token embedding.
        last_embedding = x[:, -1, :]  # (batch, embed_dim)

        # Transformer encoder blocks.
        for _ in range(self.transformer_layers):
            x = TransformerEncoderBlock(hidden_dim=self.embed_dim,
                                        num_heads=self.num_heads,
                                        dropout_rate=self.dropout_rate)(x, train=train)

        # Extract output corresponding to the last time step.
        transformer_output = x[:, -1, :]  # (batch, embed_dim)

        # Add skip connection.
        output_rep = transformer_output + last_embedding

        # MLP head.
        out = nn.Dense(self.mlp_hidden)(output_rep)
        out = nn.relu(out)
        out = nn.Dense(2 * self.output_dim)(out)

        mean, log_var = jnp.split(out, 2, axis=-1)
        var = nn.softplus(log_var) + 1e-6

        # Sample using the reparameterization trick.
        rng = self.make_rng("sample")
        epsilon = jax.random.normal(rng, mean.shape)
        sample = mean + epsilon * var

        return sample, mean, var


if __name__ == "__main__":
    print("Starting transformer belief module test...", flush=True)

    # Model hyperparameters.
    max_seq_len = 50
    state_dim = 6
    embed_dim = 64
    transformer_layers = 2
    num_heads = 4
    mlp_hidden = 128
    output_dim = 1

    dummy_input = jnp.ones((1, 10, state_dim))

    model = BeliefUpdateTransformer(
        max_seq_len=max_seq_len,
        state_dim=state_dim,
        embed_dim=embed_dim,
        transformer_layers=transformer_layers,
        num_heads=num_heads,
        mlp_hidden=mlp_hidden,
        output_dim=output_dim,
        dropout_rate=0.1
    )

    rng = jax.random.PRNGKey(0)
    # Split rng into three for params, sample, and dropout.
    rng, init_rng, sample_rng, dropout_rng = jax.random.split(rng, 4)
    variables = model.init({'params': init_rng, 'sample': sample_rng, 'dropout': dropout_rng},
                             dummy_input, train=True)

    sample_out, mean_out, var_out = model.apply(variables, dummy_input, train=True,
                                                 rngs={'sample': sample_rng, 'dropout': dropout_rng})
    print("Sample output:", np.array(sample_out), flush=True)
    print("Mean output:", np.array(mean_out), flush=True)
    print("Variance output:", np.array(var_out), flush=True)
