"""Component Layers."""
import tensorflow as tf


def msa(x: tf.Tensor, n_head: int, d_mha: int, drop_rate: float) -> tf.Tensor:
    """Multi-head Self Attention.

    Args:
        x: input tensor of shape (batch_size, len_q, d_model)
        n_head: number of heads
        d_mha: dimension of multi-head attention
        drop_rate: dropout rate

    Returns:
        output tensor of shape (batch_size, len_q, d_model)
    """
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=n_head,
        key_dim=d_mha,
        dropout=drop_rate,
    )(query=x, key=x, value=x, return_attention_scores=False)
    return tf.keras.layers.LayerNormalization()(attn_out + x)


def mca(
    x: tf.Tensor,
    ctx: tf.Tensor,
    n_head: int,
    d_mha: int,
    drop_rate: float,
) -> tf.Tensor:
    """Multi-head Cross Attention.

    Args:
        x: input tensor of shape (batch_size, len_q, d_model)
        ctx: context tensor of shape (batch_size, len_ctx, d_model)
        n_head: number of heads
        d_mha: dimension of multi-head attention
        drop_rate: dropout rate

    Returns:
        output tensor of shape (batch_size, len_q, d_model)
    """
    # TODO: get scores
    attn_out, _ = tf.keras.layers.MultiHeadAttention(
        num_heads=n_head,
        key_dim=d_mha,
        dropout=drop_rate,
    )(query=x, key=ctx, value=ctx, return_attention_scores=True)
    return tf.keras.layers.LayerNormalization()(attn_out + x)


def mmsa(x: tf.Tensor, n_head: int, d_mha: int, drop_rate: float) -> tf.Tensor:
    """Masked Multi-head Self Attention.

    Args:
        x: input tensor of shape (batch_size, len_q, d_model)
        n_head: number of heads
        d_mha: dimension of multi-head attention
        drop_rate: dropout rate

    Returns:
        output tensor of shape (batch_size, len_q, d_model)
    """
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=n_head,
        key_dim=d_mha,
        dropout=drop_rate,
    )(query=x, key=x, value=x, use_causal_mask=True)
    return tf.keras.layers.LayerNormalization()(attn_out + x)


def ff(x: tf.Tensor, d_ff: int, d_model: int, drop_rate: float) -> tf.Tensor:
    """Feed Forward.

    Args:
        x: input tensor of shape (batch_size, len_q, d_model)
        d_ff: dimension of feed forward
        d_model: embedding size
        drop_rate: dropout rate

    Returns:
        output tensor of shape (batch_size, len_q, d_model)
    """
    x = tf.keras.layers.Dense(units=d_ff, activation="relu")(x)
    x = tf.keras.layers.Dense(units=d_model)(x)
    x = tf.keras.layers.Dropout(rate=drop_rate)(x)
    return tf.keras.layers.LayerNormalization()(x + x)
