"""Encoder / Decoder of Transformer."""
import tensorflow as tf

from trans.model._layers import ff
from trans.model._layers import mca
from trans.model._layers import mmsa
from trans.model._layers import msa


def encoder(  # noqa: PLR0913
    x: tf.Tensor,
    n_head: int,
    d_mha: int,
    d_ff: int,
    d_model: int,
    drop_rate: float,
) -> tf.Tensor:
    """Encoder of Transformer.

    Args:
        x: Input tensor.
        n_head: Number of heads.
        d_mha: Dimension of multi-head attention.
        d_ff: Dimension of feed-forward layer.
        d_model: Dimension of embedding.
        drop_rate: Dropout rate.

    Returns:
        tf.Tensor: Output tensor.
    """
    x = msa(x, n_head, d_mha, drop_rate)
    return ff(x, d_ff, d_model, drop_rate)


def decoder(  # noqa: PLR0913
    x: tf.Tensor,
    ctx: tf.Tensor,
    n_head: int,
    d_mha: int,
    d_ff: int,
    d_model: int,
    drop_rate: float,
) -> tf.Tensor:
    """Decoder of Transformer.

    Args:
        x: Input tensor.
        ctx: Context tensor.
        n_head: Number of heads.
        d_mha: Dimension of multi-head attention.
        d_ff: Dimension of feed-forward layer.
        d_model: Dimension of embedding.
        drop_rate: Dropout rate.

    Returns:
        tf.Tensor: Output tensor.
    """
    x = mmsa(x, n_head, d_mha, drop_rate)
    x = mca(x, ctx, n_head, d_mha, drop_rate)
    return ff(x, d_ff, d_model, drop_rate)
