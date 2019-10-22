#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Glencoe Software, Inc. All rights reserved.
#
# This software is distributed under the terms described by the LICENSE.txt
# file you can find at the root of the distribution bundle.  If the file is
# missing please request a copy by contacting info@glencoesoftware.com

import json
import os

import numpy as np
import pixelengine
import softwarerendercontext
import softwarerenderbackend

from PIL import Image
from concurrent import futures
from math import ceil
from multiprocessing import cpu_count


class WriteTiles(object):

    def __init__(self, tile_width, tile_height, no_pyramid, input_path):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.no_pyramid = no_pyramid
        self.input_path = input_path

        self.slide_directory = input_path.replace(".isyntax", "_converted")
        os.mkdir(self.slide_directory)

        render_context = softwarerendercontext.SoftwareRenderContext()
        render_backend = softwarerenderbackend.SoftwareRenderBackend()

        self.pixel_engine = pixelengine.PixelEngine(
            render_backend, render_context
        )
        self.pixel_engine["in"].open(input_path, "ficom")

    def __enter__(self):
        return self

    def __exit__(self):
        self.pixel_engine["in"].close()

    def write_metadata(self):
        '''write metadata to a JSON file'''
        metadata_file = os.path.join(self.slide_directory, "METADATA.json")
        with open(metadata_file, "wb") as f:
            metadata = {
                "Barcode":
                    self.pixel_engine.BARCODE,
                "DICOM acquisition date":
                    self.pixel_engine.DICOM_ACQUISITION_DATETIME,
                "DICOM last calibration date":
                    self.pixel_engine.DICOM_DATE_OF_LAST_CALIBRATION,
                "DICOM time of last calibration":
                    self.pixel_engine.DICOM_TIME_OF_LAST_CALIBRATION,
                "DICOM manufacturer": self.pixel_engine.DICOM_MANUFACTURER,
                "DICOM manufacturer model name":
                    self.pixel_engine.DICOM_MANUFACTURERS_MODEL_NAME,
                "DICOM device serial number":
                    self.pixel_engine.DICOM_DEVICE_SERIAL_NUMBER,
                "Color space transform":
                    self.pixel_engine.colorspaceTransform(),
                "Block size":
                    self.pixel_engine.blockSize(),
                "Number of tiles":
                    self.pixel_engine.numTiles(),
                "Bits stored":
                    self.pixel_engine.bitsStored(),
                "Derivation description":
                    self.pixel_engine.DICOM_DERIVATION_DESCRIPTION,
                "DICOM software version":
                    self.pixel_engine.DICOM_SOFTWARE_VERSIONS,
                "Number of images":
                    self.pixel_engine.numImages()
            }

            for image in range(self.pixel_engine.numImages()):
                img = self.pixel_engine[image]
                image_metadata = {
                    "Image type": img.IMAGE_TYPE,
                    "DICOM lossy image compression method":
                        img.DICOM_LOSSY_IMAGE_COMPRESSION_METHOD,
                    "DICOM lossy image compression ratio":
                        img.DICOM_LOSSY_IMAGE_COMPRESSION_RATIO,
                    "DICOM derivation description":
                        img.DICOM_DERIVATION_DESCRIPTION,
                    "Image dimension names":
                        img.IMAGE_DIMENSION_NAMES,
                    "Image dimension types":
                        img.IMAGE_DIMENSION_TYPES,
                    "Image dimension units":
                        img.IMAGE_DIMENSION_UNITS,
                    "Image dimension ranges":
                        img.IMAGE_DIMENSION_RANGES,
                    "Image dimension discrete values":
                        img.IMAGE_DIMENSION_DISCRETE_VALUES_STRING,
                    "Image scale factor":
                        img.IMAGE_SCALE_FACTOR
                }

                if img.IMAGE_TYPE == "WSI":
                    view = self.pixel_engine.SourceView()
                    image_metadata["Bits allocated"] = view.bitsAllocated()
                    image_metadata["Bits stored"] = view.bitsStored()
                    image_metadata["High bit"] = view.highBit()
                    image_metadata["Pixel representation"] = \
                        view.pixelRepresentation()
                    image_metadata["Planar configuration"] = \
                        view.planarConfiguration()
                    image_metadata["Samples per pixel"] = \
                        view.samplesPerPixel()
                    image_metadata["Number of levels"] = \
                        self.pixel_engine.numLevels()

                metadata["Image #" + str(image)] = image_metadata

            json.dump(metadata, f)

    def write_label_image(self):
        '''write the label image (if present) as a JPEG file'''
        self.write_image_type("LABELIMAGE")

    def write_macro_image(self):
        '''write the macro image (if present) as a JPEG file'''
        self.write_image_type("MACROIMAGE")

    def find_image_type(self, image_type):
        '''look up a given image type in the pixel engine'''
        for index in range(self.pixel_engine.numImages()):
            if image_type == self.pixel_engine[index].IMAGE_TYPE:
                return self.pixel_engine[index]
        return None

    def write_image_type(self, image_type):
        '''write an image of the specified type'''
        image_container = self.find_image_type(image_type)
        if image_container is not None:
            pixels = image_container.IMAGE_DATA
            image_file = os.path.join(
                self.slide_directory, '%s.jpg' % image_type
            )
            with open(image_file, "wb") as image:
                image.write(pixels)
            print("wrote %s image" % image_type)

    def write_pyramid(self):
        '''write the slide's pyramid as a set of tiles'''
        pe_in = self.pixel_engine["in"]
        image_container = self.find_image_type(pe_in, "WSI")

        scanned_areas = image_container.IMAGE_VALID_DATA_ENVELOPES
        if scanned_areas is None:
            raise RuntimeError("No valid data envelopes")

        resolutions = range(pe_in.numLevels())
        if self.no_pyramid:
            resolutions = [0]

        source_view = pe_in.SourceView()
        for resolution in resolutions:
            # create one tile directory per resolution level
            tile_directory = os.path.join(
                self.slide_directory, str(resolution)
            )
            os.mkdir(tile_directory)

            # assemble data envelopes (== scanned areas) to extract for
            # this level
            dim_ranges = source_view.dimensionRanges(resolution)
            print("dimension ranges = " + str(dim_ranges))
            image_height = \
                (dim_ranges[1][2] - dim_ranges[1][0]) / dim_ranges[1][1]
            image_width = \
                (dim_ranges[0][2] - dim_ranges[0][0]) / dim_ranges[0][1]

            y_tiles = int(ceil(image_height / self.tile_height))
            x_tiles = int(ceil(image_width / self.tile_width))

            print("# of X tiles = %s" % x_tiles)
            print("# of Y tiles = %s" % y_tiles)

            patches = self.create_patch_list(
                [x_tiles, y_tiles], [self.tile_width, self.tile_height],
                [dim_ranges[0][0], dim_ranges[1][0]], resolution,
                tile_directory
            )

            envelopes = source_view.dataEnvelopes(resolution)
            regions = source_view.requestRegions(
                patches, envelopes, True, [0, 0, 0])

            def write_tile(pixels, width, height, filename):
                image = Image.frombuffer(
                    'RGB', (int(width), int(height)),
                    pixels, 'raw', 'RGB', 0, 1
                )
                image.save(filename)

            jobs = ()
            with futures.ThreadPoolExecutor(
                max_workers=cpu_count()
            ) as executor:
                while regions:
                    regions_ready = self.pixel_engine.waitAny(regions)
                    for region_index, region in enumerate(regions_ready):
                        view_range = region.range
                        print("processing tile %s" % view_range)
                        x_start, x_end, y_start, y_end, level = view_range
                        width = int(1 + (x_end - x_start) / dim_ranges[0][1])
                        height = int(1 + (y_end - y_start) / dim_ranges[1][1])
                        pixel_buffer_size = width * height * 3
                        pixels = np.empty(
                            int(pixel_buffer_size), dtype=np.uint8
                        )
                        region.get(pixels)
                        regions.remove(region)

                        directory = os.path.join(tile_directory, str(x_start))
                        filename = os.path.join(
                            directory, "%s.tiff" % y_start
                        )
                        jobs = jobs + (executor.submit(
                            write_tile, pixels, width, height, filename
                        ))
            futures.wait(jobs, return_when=futures.ALL_COMPLETED)

    def create_patch_list(
        self, tiles, tile_size, origin, level, tile_directory
    ):
        patches = []
        scale = 2 ** level
        tile_size[0] = tile_size[0] * scale
        tile_size[1] = tile_size[1] * scale
        for y in range(tiles[1]):
            y_start = origin[1] + (y * tile_size[1])
            y_end = (y_start + tile_size[1]) - scale
            for x in range(tiles[0]):
                x_start = origin[0] + (x * tile_size[0])
                x_end = (x_start + tile_size[0]) - scale
                patch = [x_start, x_end, y_start, y_end, level]
                patches.append(patch)

                x_directory = os.path.join(tile_directory, str(x_start))
                if not os.path.exists(x_directory):
                    os.mkdir(x_directory)
        return patches
