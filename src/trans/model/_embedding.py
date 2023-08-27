"""Input Encoding and Embedding."""
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

    # (seq, 1)
    positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    # (1, depth)
    depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    return tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)


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


def embedding(
    x: tf.Tensor,
    vocab_size: int,
    seq_len: int,
    depth: int,
) -> tf.Tensor:
    """Input Embedding.

    This function is used to encode each token in the sequence into a vector,
    and then add the positional encoding to each token.

    Args:
        x (tf.Tensor): Token ID sequence tensor of shape (batch, seq).
        vocab_size (int): Vocabulary size.
        seq_len (int): Sequence length.
        depth (int): Embedding depth.

    Returns:
        tf.Tensor: Embedding tensor of shape (batch, seq, depth).
    """
    pos_enc = wpe(seq_len, depth)
    x = wte(x, vocab_size, depth)
    return x + pos_enc
