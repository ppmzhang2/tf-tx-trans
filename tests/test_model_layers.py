"""Test `trans.model._layers.py`."""
import tensorflow as tf

from trans.model._layers import ff
from trans.model._layers import mca
from trans.model._layers import mmsa
from trans.model._layers import msa

EPS = 1e-6

B = 4
D_MODEL = 256
D_FF = D_MODEL * 4
N_HEAD = 8
D_MHA = D_MODEL // N_HEAD
LEN_Q = 8
LEN_CTX = 16
DROP_RATE = 0.1
# shapes
SHAPE_X = (B, LEN_Q, D_MODEL)
SHAPE_CTX = (B, LEN_CTX, D_MODEL)


def test_msa() -> None:
    """Test `msa` function."""
    io_shape = (B, LEN_Q, D_MODEL)
    inputs = tf.keras.Input(shape=io_shape[1:], dtype=tf.float32)
    outputs = msa(inputs, N_HEAD, D_MHA, DROP_RATE)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    x = tf.random.uniform(shape=io_shape, maxval=1, dtype=tf.float32)
    assert model(x).shape == io_shape


def test_mca() -> None:
    """Test `mca` function."""
    x_shape = (B, LEN_Q, D_MODEL)
    ctx_shape = (B, LEN_CTX, D_MODEL)
    input_x = tf.keras.Input(shape=x_shape[1:], dtype=tf.float32)
    input_ctx = tf.keras.Input(shape=ctx_shape[1:], dtype=tf.float32)
    outputs = mca(input_x, input_ctx, N_HEAD, D_MHA, DROP_RATE)
    model = tf.keras.Model(inputs=[input_x, input_ctx], outputs=outputs)
    x = tf.random.uniform(shape=x_shape, maxval=1, dtype=tf.float32)
    ctx = tf.random.uniform(shape=ctx_shape, maxval=1, dtype=tf.float32)
    assert model([x, ctx]).shape == x_shape


def test_mmsa() -> None:
    """Test `mmsa` function.

    First feed into the layer a random tensor `x`, then add to `x` a delta
    tensor, which are all zeros except for the last `LEN_DELTA` elements, and
    feed the new tensor into the layer again.
    The two outputs should be the same as for the causal masked attention the
    early sequence elements do NOT depend on later ones.
    """

    def get_delta(delta_len: int) -> tf.Tensor:
        """Get delta tensor.

        This tensor is all zeros except for the last `delta_len` elements.
        """
        zeros = tf.zeros(
            shape=(B, LEN_Q - delta_len, D_MODEL),
            dtype=tf.float32,
        )
        rand = tf.random.uniform(
            shape=(B, delta_len, D_MODEL),
            maxval=1,
            dtype=tf.float32,
        )
        return tf.concat([zeros, rand], axis=1)

    x_shape = (B, LEN_Q, D_MODEL)
    delta_len = 4
    x_shape = (B, LEN_Q, D_MODEL)
    input_x = tf.keras.Input(shape=x_shape[1:], dtype=tf.float32)
    outputs = mmsa(input_x, N_HEAD, D_MHA, DROP_RATE)
    model = tf.keras.Model(inputs=input_x, outputs=outputs)
    x1 = tf.random.uniform(shape=x_shape, maxval=1, dtype=tf.float32)
    delta = get_delta(delta_len)
    x2 = x1 + delta
    # only the first LEN_Q - delta_len elements should be the same
    o1 = model(x1, training=False)[:, :LEN_Q - delta_len, :]
    o2 = model(x2, training=False)[:, :LEN_Q - delta_len, :]
    assert tf.reduce_all(tf.abs(o1 - o2) < EPS)


def test_ff() -> None:
    """Test `ff` function."""
    io_shape = (B, LEN_Q, D_MODEL)
    inputs = tf.keras.Input(shape=io_shape[1:], dtype=tf.float32)
    outputs = ff(inputs, D_FF, D_MODEL, DROP_RATE)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    x = tf.random.uniform(shape=io_shape, maxval=1, dtype=tf.float32)
    assert model(x).shape == io_shape
