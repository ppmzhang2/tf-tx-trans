"""Component Layers."""
import tensorflow as tf


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


if __name__ == "__main__":
    B = 2
    D_MODEL = 512
    D_FF = D_MODEL * 4
    N_HEAD = 8
    D_MHA = D_MODEL // N_HEAD
    LEN_Q = 8
    LEN_CTX = 16
    SHAPE_X = (B, LEN_Q, D_MODEL)
    SHAPE_CTX = (B, LEN_CTX, D_MODEL)
    DROP_RATE = 0.1
    x = tf.random.uniform(shape=SHAPE_X, maxval=1, dtype=tf.float32)
    ctx = tf.random.uniform(shape=SHAPE_CTX, maxval=1, dtype=tf.float32)
    assert mca(x, ctx, N_HEAD, D_MHA, DROP_RATE).shape == SHAPE_X
    assert msa(x, N_HEAD, D_MHA, DROP_RATE).shape == SHAPE_X

    # validate masked multi-head self attention
    # as the output for early sequence elements doesn't depend on later ones
    inputs = tf.keras.Input(shape=SHAPE_X[1:])
    mdl = tf.keras.Model(inputs=inputs,
                         outputs=mmsa(inputs, N_HEAD, D_MHA, DROP_RATE))
    DELTA = 3

    zeros = tf.zeros(shape=(B, LEN_Q - DELTA, D_MODEL), dtype=tf.float32)
    noise = tf.random.uniform(shape=(B, DELTA, D_MODEL),
                              maxval=1,
                              dtype=tf.float32)
    x_delta = tf.concat([zeros, noise], axis=1)
    o1 = mdl.predict(x)[:, :LEN_Q - DELTA, :]
    o2 = mdl.predict(x + x_delta)[:, :LEN_Q - DELTA, :]
    assert tf.reduce_sum(tf.abs(o1 - o2)).numpy() <= 1e-6

    # validate feed forward
    assert ff(x, D_FF, D_MODEL, DROP_RATE).shape == SHAPE_X
