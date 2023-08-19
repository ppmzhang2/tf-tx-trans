"""Test `trans.data_ds` module."""
import tensorflow as tf

from trans import cfg
from trans.data import id2txt_en
from trans.data import load_train_valid


def test_id2txt_en() -> None:
    """Test `id2txt_en` function."""
    ids_en = tf.constant(
        [101, 2021, 2054, 2065, 2009, 2020, 3161, 1029, 102],
        dtype=tf.int32,
    )
    txt_en = "but what if it were active?"
    assert id2txt_en(ids_en) == txt_en


def test_load_train_valid_ragged() -> None:
    """Test `load_train_valid` function with `ragged=True`."""
    train_data, val_data, _ = load_train_valid(ragged=True)
    for pt, en, lbl in train_data.take(1):
        assert pt.shape == (cfg.BATCH_SIZE, None, None)
        assert en.shape == (cfg.BATCH_SIZE, None, None)
        assert lbl.shape == (cfg.BATCH_SIZE, None, None)
        assert tf.reduce_max(pt) <= cfg.VOCAB
        assert tf.reduce_max(en) <= cfg.VOCAB
        assert tf.reduce_max(lbl) <= cfg.VOCAB
        assert tf.reduce_min(pt) >= 0
        assert tf.reduce_min(en) >= 0
        assert tf.reduce_min(lbl) >= 0


def test_load_train_valid_padded() -> None:
    """Test `load_train_valid` function with `ragged=False`."""
    train_data, val_data, _ = load_train_valid(ragged=False)
    for pt, en, lbl in train_data.take(1):
        assert pt.shape[0] == cfg.BATCH_SIZE
        assert en.shape[0] == cfg.BATCH_SIZE
        assert lbl.shape[0] == cfg.BATCH_SIZE
        assert pt.shape[1] <= cfg.SEQ_LEN
        assert en.shape[1] <= cfg.SEQ_LEN
        assert lbl.shape[1] <= cfg.SEQ_LEN
        assert tf.reduce_max(pt) <= cfg.VOCAB
        assert tf.reduce_max(en) <= cfg.VOCAB
        assert tf.reduce_max(lbl) <= cfg.VOCAB
        assert tf.reduce_min(pt) >= 0
        assert tf.reduce_min(en) >= 0
        assert tf.reduce_min(lbl) >= 0
