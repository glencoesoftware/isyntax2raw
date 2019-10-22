[![AppVeyor status](https://ci.appveyor.com/api/projects/status/github/isyntax-to-raw)](https://ci.appveyor.com/project/gs-jenkins/isyntax-to-raw)

iSyntax Converter
=================

Python tool that uses Philips' SDK to write slides in an intermediary raw format.

Requirements
============

* Philips iSyntax SDK (https://openpathology.philips.com)
* PIL
* Numpy

Usage
=====

Basic usage is::

    python write_tiles.py /path/to/input.isyntax

Output tile width and height can optionally be specified; both are 512 by default.
A directory structure containing the pyramid tiles at all resolutions and macro/label images
will be created.  Additional metadata is written to a JSON file.  The root directory is in the same directory as the .isyntax file.
Be mindful of available disk space, as larger .isyntax files can result in >20 GB of tiles.

Areas to improve
================

* Conversion time
    - In local testing, run time varies from 1m58s to 38m36s (1.isyntax and 9.isyntax respectively)
* Disk space usage
    - Maybe makes more sense to write just the largest resolution, and have the Java code downsample during OME-TIFF creation?
        * this can now be done using the "--no_pyramid" flag
    - Compress the tiles when writing?
* Currently assumes brightfield (RGB, 8 bits per channel) without really checking the metadata.  Probably should check bit depths etc.
* Build system and packaging...
