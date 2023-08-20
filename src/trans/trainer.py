"""Model training."""
import logging
import os

import tensorflow as tf

from trans import cfg
from trans import risk
from trans.data import pten
from trans.model import get_tx_micro
from trans.model import get_tx_nano

LOGGER = logging.getLogger(__name__)

LR_INIT = 1e-8
BETA_1 = 0.9
BETA_2 = 0.98
EPSILON = 1e-9
BATCH_PER_EPOCH = pten.N_OBS_TR // cfg.BATCH_SIZE

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR_INIT,
    beta_1=BETA_1,
    beta_2=BETA_2,
    epsilon=EPSILON,
)
loss_tr = tf.keras.metrics.Mean(name="train_loss")


def get_chpt_path(*, nano: bool = False) -> str:
    """Get the checkpoint path.

    Args:
        nano (bool): Whether to use the nano or standard micro model.

    Returns:
        str: Checkpoint path.
    """
    suffix = "nano" if nano else "micro"
    return os.path.join(cfg.MODELDIR, f"ckpt_tx_{suffix}")


def lr(step: int, *, d_model: int = 512, warmup: int = 4000) -> tf.Tensor:
    """Learning rate schedule, based on the Transformer paper.

    Args:
        step (int): Current training step.
        d_model (int, optional): Model dimensionality. Defaults to 512.
        warmup (int, optional): Warmup steps. Defaults to 4000.

    Returns:
        tf.Tensor: scalar learning rate.
    """
    arg1 = tf.math.rsqrt(float(step))
    arg2 = step * (warmup**-1.5)

    return tf.math.rsqrt(float(d_model)) * tf.math.minimum(arg1, arg2)


def load_model(
    *,
    nano: bool = False,
) -> tuple[tf.keras.Model, tf.train.CheckpointManager]:
    """Load the TX model and the latest checkpoint manager.

    If no checkpoint is found, the model and the checkpoint manager are
    initialized from scratch.

    Args:
        nano (bool, optional): Whether to use the nano or standard micro model.

    Returns:
        tuple[tf.keras.Model, tf.train.CheckpointManager]: TX model and
            checkpoint manager.
    """
    mdl = get_tx_nano() if nano else get_tx_micro()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=mdl)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=get_chpt_path(nano=nano),
        max_to_keep=10,
    )

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        LOGGER.info(f"Restored from {manager.latest_checkpoint}")
    else:
        LOGGER.info("Initializing from scratch.")

    return mdl, manager


def train_step(
    model: tf.keras.Model,
    step: int,
    pt: tf.Tensor,
    en: tf.Tensor,
    lab: tf.Tensor,
) -> None:
    """Train tx model for one step.

    Args:
        model (tf.keras.Model): RPN model.
        step (int): Current training step index.
        pt (tf.Tensor): context (PT) tensor.
        en (tf.Tensor): query (EN) tensor.
        lab (tf.Tensor): label (EN) tensor.
        clip_norm (float, optional): Gradient clipping norm. Defaults to 5.0.
    """
    with tf.GradientTape() as tape:
        prd = model((en, pt), training=True)
        loss = risk.risk_cce(lab, prd)
    grads = tape.gradient(loss, model.trainable_variables)
    # check NaN
    for grad in grads:
        if tf.math.reduce_any(tf.math.is_nan(grad)):
            msg = "NaN gradient detected."
            raise ValueError(msg)
    # clip gradient
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.learning_rate = lr(step)
    optimizer.apply_gradients(
        zip(  # noqa: B905
            grads,
            model.trainable_variables,
        ))
    loss_tr(loss)


def train_tx(
    epochs: int,
    save_intv: int,
    *,
    nano: bool = False,
) -> None:
    """Train TX model.

    Args:
        epochs (int): number of epochs.
        save_intv (int): interval to save the model.
        nano (bool, optional): Whether to use the nano or standard micro model.

    Raises:
        ValueError: If NaN gradient is detected.
    """
    # Load model and checkpoint manager
    tx, manager = load_model(nano=nano)
    # Load dataset
    ds_tr, ds_va, ds_info = pten.load_train_valid(ragged=False)

    # Training loop
    for ep in range(epochs):
        # Reset the metrics at the start of the next epoch
        loss_tr.reset_states()
        for i, (pt, en, lab) in enumerate(ds_tr):
            step = ep * BATCH_PER_EPOCH + i + 1
            train_step(tx, step, pt, en, lab)
            LOGGER.info(f"Epoch {ep + 1:02d} Batch {i + 1:03d} "
                        f"Training Loss {loss_tr.result():.4f}")

            # Save model every 'save_interval' batches
            if i % save_intv == 0:
                LOGGER.info(f"Saving checkpoint for epoch {ep + 1} "
                            f"at batch {i + 1}")
                manager.save(checkpoint_number=i)
