#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Glencoe Software, Inc. All rights reserved.
#
# This software is distributed under the terms described by the LICENSE.txt
# file you can find at the root of the distribution bundle.  If the file is
# missing please request a copy by contacting info@glencoesoftware.com

import argparse
import json
import os
from concurrent import futures
from math import ceil
from multiprocessing import cpu_count
import numpy as np
from PIL import Image
import pixelengine
import softwarerendercontext
import softwarerenderbackend

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="path to isyntax file")
    parser.add_argument("--tile_width", default=512, type=int, help="tile width in pixels")
    parser.add_argument("--tile_height", default=512, type=int, help="tile height in pixels")
    parser.add_argument("--no_pyramid", help="toggle subresolution writing", action="store_true")
    args = parser.parse_args()
    input_file = args.input_path
    tile_width = args.tile_width
    tile_height = args.tile_height
    no_pyramid = args.no_pyramid

    slide_directory, pe_in, pixel_engine = setup(input_file)
    write_metadata(slide_directory, pe_in)
    write_label_image(slide_directory, pe_in)
    write_macro_image(slide_directory, pe_in)
    write_pyramid(slide_directory, pe_in, pixel_engine, tile_width, tile_height, no_pyramid)
    pe_in.close()


# set up the pixel engine and output directory structure
def setup(input_file):
    slide_dir = input_file.replace(".isyntax", "_converted")
    os.mkdir(slide_dir)

    render_context = softwarerendercontext.SoftwareRenderContext()
    render_backend = softwarerenderbackend.SoftwareRenderBackend()

    pixel_engine = pixelengine.PixelEngine(render_backend, render_context)
    pixel_engine["in"].open(input_file, "ficom")
    return slide_dir, pixel_engine["in"], pixel_engine

# write metadata to a JSON file
def write_metadata(slide_directory, pixel_engine):
    metadata_file = open(slide_directory + os.sep + "METADATA.json", "wb")

    metadata = dict()
    metadata["Barcode"] = pixel_engine.BARCODE
    metadata["DICOM acquisition date"] = pixel_engine.DICOM_ACQUISITION_DATETIME
    metadata["DICOM last calibration date"] = pixel_engine.DICOM_DATE_OF_LAST_CALIBRATION
    metadata["DICOM time of last calibration"] = pixel_engine.DICOM_TIME_OF_LAST_CALIBRATION
    metadata["DICOM manufacturer"] = pixel_engine.DICOM_MANUFACTURER
    metadata["DICOM manufacturer model name"] = pixel_engine.DICOM_MANUFACTURERS_MODEL_NAME
    metadata["DICOM device serial number"] = pixel_engine.DICOM_DEVICE_SERIAL_NUMBER
    metadata["Color space transform"] = pixel_engine.colorspaceTransform()
    metadata["Block size"] = pixel_engine.blockSize()
    metadata["Number of tiles"] = pixel_engine.numTiles()
    metadata["Bits stored"] = pixel_engine.bitsStored()
    metadata["Derivation description"] = pixel_engine.DICOM_DERIVATION_DESCRIPTION
    metadata["DICOM software version"]= pixel_engine.DICOM_SOFTWARE_VERSIONS
    metadata["Number of images"] = pixel_engine.numImages()

    for image in range(pixel_engine.numImages()):
        image_metadata = dict()
        image_metadata["Image type"] = pixel_engine[image].IMAGE_TYPE
        image_metadata["DICOM lossy image compression method"] = pixel_engine[image].DICOM_LOSSY_IMAGE_COMPRESSION_METHOD
        image_metadata["DICOM lossy image compression ratio"] = pixel_engine[image].DICOM_LOSSY_IMAGE_COMPRESSION_RATIO
        image_metadata["DICOM derivation description"] = pixel_engine[image].DICOM_DERIVATION_DESCRIPTION
        image_metadata["Image dimension names"] = pixel_engine[image].IMAGE_DIMENSION_NAMES
        image_metadata["Image dimension types"] = pixel_engine[image].IMAGE_DIMENSION_TYPES
        image_metadata["Image dimension units"] = pixel_engine[image].IMAGE_DIMENSION_UNITS
        image_metadata["Image dimension ranges"] = pixel_engine[image].IMAGE_DIMENSION_RANGES
        image_metadata["Image dimension discrete values"] = pixel_engine[image].IMAGE_DIMENSION_DISCRETE_VALUES_STRING
        image_metadata["Image scale factor"] = pixel_engine[image].IMAGE_SCALE_FACTOR

        if pixel_engine[image].IMAGE_TYPE == "WSI":
            view = pixel_engine.SourceView()
            image_metadata["Bits allocated"] = view.bitsAllocated()
            image_metadata["Bits stored"] = view.bitsStored()
            image_metadata["High bit"] = view.highBit()
            image_metadata["Pixel representation"] = view.pixelRepresentation()
            image_metadata["Planar configuration"] = view.planarConfiguration()
            image_metadata["Samples per pixel"] = view.samplesPerPixel()
            image_metadata["Number of levels"] = pixel_engine.numLevels()

        metadata["Image #" + str(image)] = image_metadata

    json.dump(metadata, metadata_file)

# write the label image (if present) as a JPEG file
def write_label_image(slide_directory, pixel_engine):
    write_image_type(slide_directory, pixel_engine, "LABELIMAGE")

# write the macro image (if present) as a JPEG file
def write_macro_image(slide_directory, pixel_engine):
    write_image_type(slide_directory, pixel_engine, "MACROIMAGE")

# write the slide's pyramid as a set of tiles
def write_pyramid(slide_directory, pe_in, pixel_engine, tile_width, tile_height, no_pyramid):
    image_container = find_image_type(pe_in, "WSI")

    scanned_areas = image_container.IMAGE_VALID_DATA_ENVELOPES
    if scanned_areas == None:
        raise RuntimeError("No valid data envelopes")

    source_view = pe_in.SourceView()

    resolutions = range(pe_in.numLevels())
    if no_pyramid:
        resolutions = [0]

    for resolution in resolutions:
        # create one tile directory per resolution level
        tile_directory = slide_directory + os.sep + str(resolution)
        os.mkdir(tile_directory)

        # assemble data envelopes (== scanned areas) to extract for this level

        dim_ranges = source_view.dimensionRanges(resolution)
        print("dimension ranges = " + str(dim_ranges))
        image_height = (dim_ranges[1][2] - dim_ranges[1][0]) / dim_ranges[1][1]
        image_width = (dim_ranges[0][2] - dim_ranges[0][0]) / dim_ranges[0][1]

        y_tiles = int(ceil(image_height / float(tile_height)))
        x_tiles = int(ceil(image_width / float(tile_width)))

        print("# of X tiles = " + str(x_tiles))
        print("# of Y tiles = " + str(y_tiles))

        patches = create_patch_list([x_tiles, y_tiles], [tile_width, tile_height], [dim_ranges[0][0], dim_ranges[1][0]], resolution, tile_directory)

        envelopes = source_view.dataEnvelopes(resolution)
        regions = source_view.requestRegions(patches, envelopes, True, [0, 0, 0])

        jobs = ()

        with futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            while regions:
                regions_ready = pixel_engine.waitAny(regions)
                for region_index, region in enumerate(regions_ready):
                    view_range = region.range
                    print("processing tile " + str(view_range))
                    x_start, x_end, y_start, y_end, level = view_range
                    width = int(1 + (x_end - x_start) / dim_ranges[0][1])
                    height = int(1 + (y_end - y_start) / dim_ranges[1][1])
                    pixel_buffer_size = width * height * 3
                    pixels = np.empty(int(pixel_buffer_size), dtype=np.uint8)
                    region.get(pixels)
                    regions.remove(region)

                    directory = tile_directory + os.sep + str(x_start)
                    file_name = directory + os.sep + str(y_start) + ".tiff"
                    jobs = jobs + (executor.submit(write_tile, pixels, width, height, file_name),)
        futures.wait(jobs, return_when=futures.ALL_COMPLETED)

def write_tile(pixels, width, height, file_name):
    image = Image.frombuffer('RGB', (int(width), int(height)), pixels, 'raw', 'RGB', 0, 1);
    image.save(file_name)

def create_patch_list(tiles, tile_size, origin, level, tile_directory):
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

      x_directory = tile_directory + os.sep + str(x_start)
      if not os.path.exists(x_directory):
          os.mkdir(x_directory)

  return patches

# write an image of the specified type
def write_image_type(slide_directory, pixel_engine, image_type):
    image_container = find_image_type(pixel_engine, image_type)
    if image_container != None:
        pixels = image_container.IMAGE_DATA
        image = open(slide_directory + os.sep + image_type + ".jpg", "wb")
        image.write(pixels)
        print("wrote " + image_type + " image")

# look up a given image type in the pixel engine
def find_image_type(pixel_engine, image_type):
    for index in range(pixel_engine.numImages()):
        if image_type == pixel_engine[index].IMAGE_TYPE:
            return pixel_engine[index]
    return None

if __name__ == '__main__':
    main()
