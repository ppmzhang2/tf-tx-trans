"""Transformer Model."""
import tensorflow as tf

from trans import cfg
from trans.model._embedding import embedding
from trans.model._encde import decoder
from trans.model._encde import encoder


def tx(x: tf.Tensor, ctx: tf.Tensor) -> tf.Tensor:
    """Transformer Model.

    Args:
        x: Input tensor.
        ctx: Context tensor.

    Returns:
        Output tensor.
    """
    ctx = embedding(ctx, cfg.VOCAB_SIZE, cfg.D_MODEL)
    ctx = tf.keras.layers.Dropout(cfg.DROP_RATE)(ctx)

    for _ in range(cfg.N_LAYER):
        ctx = encoder(
            ctx,
            cfg.N_HEAD,
            cfg.D_MHA,
            cfg.D_FF,
            cfg.D_MODEL,
            cfg.DROP_RATE,
        )

    x = embedding(x, cfg.VOCAB_SIZE, cfg.D_MODEL)
    x = tf.keras.layers.Dropout(cfg.DROP_RATE)(x)

    for _ in range(cfg.N_LAYER):
        x = decoder(
            x,
            ctx,
            cfg.N_HEAD,
            cfg.D_MHA,
            cfg.D_FF,
            cfg.D_MODEL,
            cfg.DROP_RATE,
        )

    return tf.keras.layers.Dense(cfg.VOCAB_SIZE)(x)
