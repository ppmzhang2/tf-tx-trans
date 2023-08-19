"""Input Encoding and Embedding."""
import numpy as np
import tensorflow as tf


def wpe(length: int, depth: int) -> tf.Tensor:
    """Word Positional Encoding.

    This function is used to encode the position of each token in the sequence,
    which is then added to the embedding of each token.

    Args:
        length: Sequence length.
        depth: Embedding depth.

    Returns:
        Positional encoding tensor of shape (length, depth).
    """
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


def wte(x: tf.Tensor, vocab_size: int, depth: int) -> tf.Tensor:
    """Word Token Embedding.

    This function is used to encode each token in the sequence into a vector.

    Args:
        x: Token ID sequence tensor of shape (batch, seq).
        vocab_size: Vocabulary size.
        depth: Embedding depth.

    Returns:
        tf.Tensor: Embedding tensor of shape (batch, seq, depth).
    """
    return tf.keras.layers.Embedding(vocab_size, depth)(x)


def embedding(x: tf.Tensor, vocab_size: int, depth: int) -> tf.Tensor:
    """Input Embedding.

    This function is used to encode each token in the sequence into a vector,
    and then add the positional encoding to each token.

    Args:
        x: Token ID sequence tensor of shape (batch, seq).
        vocab_size: Vocabulary size.
        depth: Embedding depth.

    Returns:
        tf.Tensor: Embedding tensor of shape (batch, seq, depth).
    """
    seq_len = tf.shape(x)[1]
    pos_enc = wpe(seq_len, depth)
    x = wte(x, vocab_size, depth)
    return x + pos_enc


if __name__ == '__main__':
    pos_encoding = wpe(50, 4)
    print(pos_encoding)

    x = tf.random.uniform(shape=(64, 40), maxval=31000, dtype=tf.int32)
    y = embedding(x, 31000, 512)
    print(y)
