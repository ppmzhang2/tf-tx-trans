"""Test trans.model._embedding.py."""
import tensorflow as tf

from trans.model._embedding import embedding
from trans.model._embedding import wpe

EPS = 1e-6
B = 64
SEQ_LEN = 40
D_MODEL = 512
VOCAB_SIZE = 31000


def test_embedding() -> None:
    """Test embedding function as a model."""
    inputs = tf.keras.Input(shape=(SEQ_LEN, ), dtype=tf.int32)
    outputs = embedding(inputs, VOCAB_SIZE, D_MODEL)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    in_shape = (B, SEQ_LEN)
    out_shape = (B, SEQ_LEN, D_MODEL)
    x = tf.random.uniform(shape=in_shape, maxval=VOCAB_SIZE, dtype=tf.int32)
    assert model(x).shape == out_shape


def test_wpe() -> None:
    """Test wpe function."""
    pos_enc = wpe(SEQ_LEN, D_MODEL)
    assert abs(tf.reduce_max(pos_enc).numpy() - 1.0) < EPS
    assert abs(tf.reduce_min(pos_enc).numpy() + 1.0) < EPS
    assert pos_enc.shape == (SEQ_LEN, D_MODEL)
