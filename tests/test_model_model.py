"""Test `trans.model._model.py`."""
import tensorflow as tf

from trans import cfg
from trans.model import get_tx_micro
from trans.model import get_tx_nano

B = 4


def test_get_tx() -> None:
    """Test `get_tx_micro` and `get_tx_nano` functions."""
    len_seq_en = 10
    len_seq_pt = 20
    out_shape = (B, len_seq_en, cfg.D_LABEL)
    x_en = tf.random.uniform(
        shape=(B, len_seq_en),
        minval=0,
        maxval=cfg.VOCAB,
        dtype=tf.int32,
    )
    x_pt = tf.random.uniform(
        shape=(B, len_seq_pt),
        minval=0,
        maxval=cfg.VOCAB,
        dtype=tf.int32,
    )
    tx_micro = get_tx_micro()
    tx_nano = get_tx_nano()

    assert tx_micro((x_en, x_pt), training=False).shape == out_shape
    assert tx_nano((x_en, x_pt), training=False).shape == out_shape
