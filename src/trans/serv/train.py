"""The training script for the model."""
import click

from trans import trainer


@click.command()
@click.option("--epochs",
              type=click.INT,
              default=5,
              help="number of epochs to train.")
@click.option("--save-intv",
              type=click.INT,
              default=10,
              help="number of batches between each save.")
def train_tx_micro(epochs: int, save_intv: int) -> None:
    """Train the tx-micro model with specs defined in the original paper."""
    return trainer.train_tx(epochs, save_intv, nano=False)


@click.command()
@click.option("--epochs",
              type=click.INT,
              default=5,
              help="number of epochs to train.")
@click.option("--save-intv",
              type=click.INT,
              default=10,
              help="number of batches between each save.")
def train_tx_nano(epochs: int, save_intv: int) -> None:
    """Train the tx-nano model."""
    return trainer.train_tx(epochs, save_intv, nano=True)
