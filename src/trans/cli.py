"""All CLI commands are defined here."""
import click

from trans.serv import train as train_cli


@click.group()
def cli() -> None:
    """CLI for the TX model."""


cli.add_command(train_cli.train_tx_nano)
cli.add_command(train_cli.train_tx_micro)
