"""Transformer Model."""
import tensorflow as tf

from trans import cfg
from trans.model._embedding import embedding
from trans.model._encde import decoder
from trans.model._encde import encoder


def tx_func(  # noqa: PLR0913
    x: tf.Tensor,
    ctx: tf.Tensor,
    vocab_size: int,
    n_layer: int,
    n_head: int,
    d_model: int,
    d_mha: int,
    d_ff: int,
    d_label: int,
    drop_rate: float,
) -> tf.Tensor:
    """Transformer Model.

    Args:
        x: Input tensor (English / translated text) of shape (B, LEN_X).
        ctx: Input tensor (Portuguese / original text) of shape (B, LEN_CTX).
        vocab_size: Vocabulary size.
        n_layer: Number of layers.
        n_head: Number of heads in multi-head attention.
        d_model: Model dimension.
        d_mha: Multi-head attention dimension.
        d_ff: Feed-forward dimension.
        d_label: Label dimension (vocab_size) for output layer.
        drop_rate: Dropout rate.

    Returns:
        tf.Tensor: translated (EN) text of shape (B, LEN_X, d_label).
    """
    ctx = embedding(ctx, vocab_size, d_model)
    ctx = tf.keras.layers.Dropout(drop_rate)(ctx)

    for _ in range(n_layer):
        ctx = encoder(ctx, n_head, d_mha, d_ff, d_model, drop_rate)

    x = embedding(x, vocab_size, d_model)
    x = tf.keras.layers.Dropout(drop_rate)(x)

    for _ in range(n_layer):
        x = decoder(x, ctx, n_head, d_mha, d_ff, d_model, drop_rate)

    return tf.keras.layers.Dense(d_label, activation="softmax")(x)


def get_tx_micro() -> tf.keras.Model:
    """Get tx-micro transformer model.

    Returns:
        tf.keras.Model: the micro transformer model, whose specs stick to the
            original paper.
    """
    input_x = tf.keras.Input(shape=(None, ), dtype=tf.int32, name="x")
    input_ctx = tf.keras.Input(shape=(None, ), dtype=tf.int32, name="ctx")
    out = tx_func(
        input_x,
        input_ctx,
        cfg.VOCAB,
        cfg.N_LAYER,
        cfg.N_HEAD,
        cfg.D_MODEL,
        cfg.D_MHA,
        cfg.D_FF,
        cfg.D_LABEL,
        cfg.DROP_RATE,
    )
    return tf.keras.Model(inputs=[input_x, input_ctx], outputs=out)


def get_tx_nano() -> tf.keras.Model:
    """Get tx-nano transformer model.

    Returns:
        tf.keras.Model: the minimal nano transformer model for fast validation.
    """
    input_x = tf.keras.Input(shape=(None, ), dtype=tf.int32, name="x")
    input_ctx = tf.keras.Input(shape=(None, ), dtype=tf.int32, name="ctx")
    out = tx_func(
        input_x,
        input_ctx,
        cfg.VOCAB,
        cfg.N_LAYER_NANO,
        cfg.N_HEAD_NANO,
        cfg.D_MODEL_NANO,
        cfg.D_MHA_NANO,
        cfg.D_FF_NANO,
        cfg.D_LABEL,
        cfg.DROP_RATE,
    )
    return tf.keras.Model(inputs=[input_x, input_ctx], outputs=out)
