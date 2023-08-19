"""Risk / metrics for training and evaluation."""
import tensorflow as tf


def risk_cce(lab: tf.Tensor, prd: tf.Tensor) -> tf.Tensor:
    """Compute the cross-entropy loss.

    Use sparse categorical cross-entropy as the label is of integer type, NOT
    one-hot encoded.

    Args:
        lab (tf.Tensor): label tensor (EN) of type int32 and shape (B, SEQ)
        prd (tf.Tensor): prediction tensor (EN) of shape (B, SEQ, VOCAB)

    Returns:
        tf.Tensor: scalar loss tensor
    """
    mask = tf.cast(lab != 0, dtype=tf.float32)
    cce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )
    loss = cce(lab, prd) * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def accuracy(lab: tf.Tensor, prd: tf.Tensor) -> tf.Tensor:
    """Compute the accuracy.

    Args:
        lab (tf.Tensor): label tensor (EN) of type int32 and shape (B, SEQ)
        prd (tf.Tensor): prediction tensor (EN) of shape (B, SEQ, VOCAB)

    Returns:
        tf.Tensor: scalar accuracy tensor
    """
    mask = tf.cast(lab != 0, dtype=tf.float32)
    prd_id = tf.argmax(prd, axis=-1, output_type=tf.int32)
    lab = tf.cast(lab, dtype=tf.int32)
    match = tf.cast(lab == prd_id, dtype=tf.float32) * mask
    return tf.reduce_sum(match) / tf.reduce_sum(mask)
