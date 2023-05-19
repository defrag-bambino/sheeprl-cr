"""Adapted from https://github.com/Lightning-Universe/lightning-flash/blob/master/src/flash/__main__.py"""

import functools
import importlib
import os
import warnings
from contextlib import closing
from typing import Optional
from unittest.mock import patch

import click
from lightning.fabric.fabric import _is_using_cli

from sheeprl.utils.registry import decoupled_tasks, tasks

CONTEXT_SETTINGS = dict(help_option_names=["--sheeprl_help"])


@click.group(no_args_is_help=True, add_help_option=True, context_settings=CONTEXT_SETTINGS)
def run():
    """SheepRL zero-code command line utility."""
    if not _is_using_cli():
        warnings.warn(
            "This script was launched without the Lightning CLI. Consider to launch the script with "
            "`lightning run model ...` to scale it with Fabric"
        )


def register_command(command, task, name: Optional[str] = None):
    @run.command(
        name if name is not None else command.__name__,
        context_settings=dict(
            help_option_names=[],
            ignore_unknown_options=True,
        ),
    )
    @click.argument("cli_args", nargs=-1, type=click.UNPROCESSED)
    @functools.wraps(command)
    def wrapper(cli_args):
        with patch("sys.argv", [task.__file__] + list(cli_args)) as sys_argv_mock:
            strategy = os.environ.get("LT_STRATEGY", None)
            is_cli_being_used = _is_using_cli()
            is_decoupled = name in decoupled_tasks
            if strategy == "fsdp":
                raise ValueError(
                    "FSDPStrategy is currently not supported. Please launch the script with another strategy: "
                    "`lightning run model --strategy=... sheeprl.py ...`"
                )
            if is_decoupled and not is_cli_being_used:
                import torch.distributed.run as torchrun
                from torch.distributed.elastic.utils import get_socket_with_port

                sock = get_socket_with_port()
                with closing(sock):
                    master_port = sock.getsockname()[1]
                devices = os.environ.get("LT_DEVICES")
                nproc_per_node = "2" if devices is None else devices
                torchrun_args = [
                    f"--nproc_per_node={nproc_per_node}",
                    "--nnodes=1",
                    "--node-rank=0",
                    "--start-method=spawn",
                    "--master-addr=localhost",
                    f"--master-port={master_port}",
                ] + sys_argv_mock
                torchrun.main(torchrun_args)
            elif is_decoupled and strategy is not None:
                raise ValueError(
                    f"The strategy flag has been set with value `{strategy}`: "
                    "when running decoupled algorithms with the Lightning CLI one must not set the strategy."
                )
            else:
                if not is_cli_being_used:
                    devices = os.environ.get("LT_DEVICES")
                    if devices is None:
                        os.environ["LT_DEVICES"] = "1"
                command()


for module, algos in tasks.items():
    for algo in algos:
        try:
            algo_name = algo
            task = importlib.import_module(f"sheeprl.algos.{module}.{algo_name}")

            for command in task.__all__:
                command = task.__dict__[command]
                register_command(command, task, name=algo_name)
        except ImportError:
            pass
