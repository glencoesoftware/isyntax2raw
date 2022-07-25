[![AppVeyor status](https://ci.appveyor.com/api/projects/status/github/isyntax2raw)](https://ci.appveyor.com/project/gs-jenkins/isyntax2raw)

# iSyntax Converter

Python tool that uses Philips' SDK to write slides in an intermediary raw format.

## Requirements

* Python 3.6+
* Philips iSyntax SDK (https://www.openpathology.philips.com)

The iSyntax SDK __must__ be downloaded separately from Philips and the
relevant license agreement agreed to before any conversion can take place.

As of version 0.4.0, which has a Python 3.6+ requirement, the supported
iSyntax SDK versions and environments are as follows:

* iSyntax SDK 1.2.1 (CentOS 7, Ubuntu 18.04, Windows 10 64-bit)
* iSyntax SDK 2.0 (CentOS 8, Ubuntu 18.04, Windows 10 64-bit)

## Usage

Basic usage is:

    isyntax2raw write_tiles /path/to/input.isyntax /path/to/directory.zarr

Please see `isyntax2raw write_tiles --help` for detailed information.

Output tile width and height can optionally be specified; default values are
detailed in `--help`.

A directory structure containing the pyramid tiles at all resolutions and
macro/label images will be created.  The default format is Zarr compliant with https://ngff.openmicroscopy.org/0.4.
Additional metadata is written to a JSON file.  Be mindful of available disk space, as
larger .isyntax files can result in >20 GB of tiles.

Use of the Zarr file type will result in losslessly compressed output.  This
is the only format currently supported by the downstream `raw2ometiff` (as of
version 0.3.0).

## Background color

Any missing tiles are filled with 0 by default, which displays as black.
The fill value can be changed using the `--fill_color` option, which accepts
a single integer between 0 and 255 inclusive.  Setting `--fill_color=255`
will cause any missing tiles to display as white.

## Performance

This package is __highly__ sensitive to underlying hardware as well as
the following configuration options:

 * `--max_workers`
 * `--tile_width`
 * `--tile_height`
 * `--batch_size`

On systems with significant I/O bandwidth, particularly SATA or
NVMe based storage, we have found sharply diminishing returns with worker
counts > 4.  There are significant performance gains to be had utilizing
larger tile sizes but be mindful of the consequences on the downstream
workflow.  You may find increasing the batch size on systems with very
high single core performance to give modest performance gains.

In general, expect to need to tune the above settings and measure
relative performance.

## License

The iSyntax converter is distributed under the terms of the BSD license.
Please see `LICENSE.txt` for further details.

## Areas to improve

* Currently assumes brightfield (RGB, 8 bits per channel) without really
  checking the metadata.  Probably should check bit depths etc.
