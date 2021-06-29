# encoding: utf-8
#
# Copyright (c) 2019 Glencoe Software, Inc. All rights reserved.
#
# This software is distributed under the terms described by the LICENCE file
# you can find at the root of the distribution bundle.
# If the file is missing please request a copy by contacting
# support@glencoesoftware.com.

import click
import logging

from .. import WriteTiles


def setup_logging(debug):
    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s [%(name)16s] "
               "(%(thread)10s) %(message)s"
    )


@click.group()
def cli():
    pass


@cli.command(name='write_tiles')
@click.option(
    "--tile_width", default=512, type=int, show_default=True,
    help="tile width in pixels"
)
@click.option(
    "--tile_height", default=512, type=int, show_default=True,
    help="tile height in pixels"
)
@click.option(
    "--resolutions", type=int,
    help="number of pyramid resolutions to generate [default: all]"
)
@click.option(
    "--max_workers", default=4, type=int,
    show_default=True,
    help="maximum number of tile workers that will run at one time",
)
@click.option(
    "--batch_size", default=250, type=int, show_default=True,
    help="number of patches fed into the iSyntax SDK at one time"
)
@click.option(
    "--fill_color", type=click.IntRange(min=0, max=255), default=0,
    show_default=True,
    help="background color for missing tiles (0-255)"
)
@click.option(
    "--nested/--no-nested", default=True, show_default=True,
    help="Whether to use '/' as the chunk path separator"
)
@click.option(
    "--debug", is_flag=True,
    help="enable debugging",
)
@click.argument("input_path")
@click.argument("output_path")
def write_tiles(
    tile_width, tile_height, resolutions, max_workers, batch_size,
    fill_color, nested, debug, input_path, output_path
):
    setup_logging(debug)
    with WriteTiles(
        tile_width, tile_height, resolutions, max_workers,
        batch_size, fill_color, nested, input_path, output_path
    ) as wt:
        wt.write_metadata()
        wt.write_label_image()
        wt.write_macro_image()
        wt.write_pyramid()


@cli.command(name='write_metadata')
@click.option(
    "--debug", is_flag=True,
    help="enable debugging",
)
@click.argument('input_path')
@click.argument('output_file')
def write_metadata(debug, input_path, output_file):
    setup_logging(debug)
    with WriteTiles(
        None, None, None, None,
        None, None, None, input_path, None
    ) as wt:
        wt.write_metadata_json(output_file)


def main():
    cli()
