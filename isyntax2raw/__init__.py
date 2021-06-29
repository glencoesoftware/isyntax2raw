#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Glencoe Software, Inc. All rights reserved.
#
# This software is distributed under the terms described by the LICENSE.txt
# file you can find at the root of the distribution bundle.  If the file is
# missing please request a copy by contacting info@glencoesoftware.com

from io import BytesIO
import json
import logging
import math
import os

import numpy as np
import pixelengine
import softwarerendercontext
import softwarerenderbackend
import zarr

from datetime import datetime
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from threading import BoundedSemaphore

from PIL import Image
from kajiki import PackageLoader
from zarr.storage import FSStore


log = logging.getLogger(__name__)

# version of the Zarr layout
LAYOUT_VERSION = 3


class MaxQueuePool(object):
    """This Class wraps a concurrent.futures.Executor
    limiting the size of its task queue.
    If `max_queue_size` tasks are submitted, the next call to submit will
    block until a previously submitted one is completed.

    Brought in from:
      * https://gist.github.com/noxdafox/4150eff0059ea43f6adbdd66e5d5e87e

    See also:
      * https://www.bettercodebytes.com/
            theadpoolexecutor-with-a-bounded-queue-in-python/
      * https://pypi.org/project/bounded-pool-executor/
      * https://bugs.python.org/issue14119
      * https://bugs.python.org/issue29595
      * https://github.com/python/cpython/pull/143
    """
    def __init__(self, executor, max_queue_size, max_workers=None):
        if max_workers is None:
            max_workers = max_queue_size
        self.pool = executor(max_workers=max_workers)
        self.pool_queue = BoundedSemaphore(max_queue_size)

    def submit(self, function, *args, **kwargs):
        """Submits a new task to the pool, blocks if Pool queue is full."""
        self.pool_queue.acquire()

        future = self.pool.submit(function, *args, **kwargs)
        future.add_done_callback(self.pool_queue_callback)

        return future

    def pool_queue_callback(self, _):
        """Called once task is done, releases one queue slot."""
        self.pool_queue.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.pool.__exit__(exception_type, exception_value, traceback)


class WriteTiles(object):

    def __init__(
        self, tile_width, tile_height, resolutions, max_workers,
        batch_size, fill_color, nested, input_path, output_path
    ):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.resolutions = resolutions
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.fill_color = fill_color
        self.nested = nested
        self.input_path = input_path
        self.slide_directory = output_path

        render_context = softwarerendercontext.SoftwareRenderContext()
        render_backend = softwarerenderbackend.SoftwareRenderBackend()

        self.pixel_engine = pixelengine.PixelEngine(
            render_backend, render_context
        )
        self.pixel_engine["in"].open(input_path, "ficom")
        self.sdk_v1 = hasattr(self.pixel_engine["in"], "BARCODE")

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.pixel_engine["in"].close()

    def get_metadata(self):
        if self.sdk_v1:
            return self.get_metadata_sdk_v1()
        else:
            return self.get_metadata_sdk_v2()

    def get_metadata_sdk_v1(self):
        pe_in = self.pixel_engine["in"]
        return {
            "Barcode":
                self.barcode(),
            "DICOM acquisition date":
                self.acquisition_datetime().isoformat(),
            "DICOM last calibration date":
                pe_in.DICOM_DATE_OF_LAST_CALIBRATION,
            "DICOM time of last calibration":
                pe_in.DICOM_TIME_OF_LAST_CALIBRATION,
            "DICOM manufacturer":
            pe_in.DICOM_MANUFACTURER,
            "DICOM manufacturer model name":
                pe_in.DICOM_MANUFACTURERS_MODEL_NAME,
            "DICOM device serial number":
                pe_in.DICOM_DEVICE_SERIAL_NUMBER,
            "Color space transform":
                pe_in.colorspaceTransform(),
            "Block size":
                pe_in.blockSize(),
            "Number of tiles":
                pe_in.numTiles(),
            "Bits stored":
                pe_in.bitsStored(),
            "Derivation description":
                self.derivation_description(),
            "DICOM software version":
                pe_in.DICOM_SOFTWARE_VERSIONS,
            "Number of images": self.num_images()
        }

    def get_metadata_sdk_v2(self):
        pe_in = self.pixel_engine["in"]
        return {
            "Pixel engine version":
                self.pixel_engine.version,
            "Barcode":
                self.barcode(),
            "Acquisition datetime":
                self.acquisition_datetime().isoformat(),
            "Date of last calibration":
                pe_in.date_of_last_calibration,
            "Time of last calibration":
                pe_in.time_of_last_calibration,
            "Manufacturer":
                pe_in.manufacturer,
            "Model name":
                pe_in.model_name,
            "Device serial number":
                pe_in.device_serial_number,
            "Derivation description":
                self.derivation_description(),
            "Software versions":
                pe_in.software_versions,
            "Number of images":
                self.num_images(),
            "Scanner calibration status":
                pe_in.scanner_calibration_status,
            "Scanner operator ID":
                pe_in.scanner_operator_id,
            "Scanner rack number":
                pe_in.scanner_rack_number,
            "Scanner rack priority":
                pe_in.scanner_rack_priority,
            "Scanner slot number":
                pe_in.scanner_slot_number,
            "iSyntax file version":
                pe_in.isyntax_file_version,
            # Could also add: 'is_UFS', 'is_UFSb', 'is_UVS', 'is_philips'
        }

    def get_image_metadata(self, image_no):
        if self.sdk_v1:
            return self.get_image_metadata_sdk_v1(image_no)
        else:
            return self.get_image_metadata_sdk_v2(image_no)

    def get_image_metadata_sdk_v1(self, image_no):
        pe_in = self.pixel_engine["in"]
        img = pe_in[image_no]
        image_type = self.image_type(image_no)
        image_metadata = {
            "Image type":
                image_type,
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
        if image_type == "WSI":
            self.pixel_size_x = img.IMAGE_SCALE_FACTOR[0]
            self.pixel_size_y = img.IMAGE_SCALE_FACTOR[1]

            view = pe_in.SourceView()
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
                pe_in.numLevels()

            for resolution in range(pe_in.numLevels()):
                dim_ranges = view.dimensionRanges(resolution)
                level_size_x = self.get_size(dim_ranges[0])
                level_size_y = self.get_size(dim_ranges[1])
                image_metadata["Level sizes #%s" % resolution] = {
                    "X": level_size_x,
                    "Y": level_size_y
                }
                if resolution == 0:
                    self.size_x = level_size_x
                    self.size_y = level_size_y
        elif image_type == "LABELIMAGE":
            self.label_x = self.get_size(img.IMAGE_DIMENSION_RANGES[0]) + 1
            self.label_y = self.get_size(img.IMAGE_DIMENSION_RANGES[1]) + 1
        elif image_type == "MACROIMAGE":
            self.macro_x = self.get_size(img.IMAGE_DIMENSION_RANGES[0]) + 1
            self.macro_y = self.get_size(img.IMAGE_DIMENSION_RANGES[1]) + 1
        return image_metadata

    def get_image_metadata_sdk_v2(self, image_no):
        pe_in = self.pixel_engine["in"]
        img = pe_in[image_no]
        image_type = self.image_type(image_no)
        view = img.source_view
        image_scale_factor = view.scale
        image_metadata = {
            "Image type":
                image_type,
            "Lossy image compression method":
                img.lossy_image_compression_method,
            "Lossy image compression ratio":
                img.lossy_image_compression_ratio,
            "Image dimension names":
                view.dimension_names,
            "Image dimension types":
                view.dimension_types,
            "Image dimension units":
                view.dimension_units,
            "Image dimension discrete values":
                view.dimension_discrete_values,
            "Image scale factor":
                image_scale_factor,
            "Block size":
                img.block_size(),
        }
        if image_type == "WSI":
            image_metadata["Color space transform"] = \
                img.colorspace_transform
            image_metadata["Number of tiles"] = img.num_tiles

            self.pixel_size_x = image_scale_factor[0]
            self.pixel_size_y = image_scale_factor[1]

            image_metadata["Bits allocated"] = view.bits_allocated
            image_metadata["Bits stored"] = view.bits_stored
            image_metadata["High bit"] = view.high_bit
            image_metadata["Pixel representation"] = \
                view.pixel_representation
            image_metadata["Planar configuration"] = \
                view.planar_configuration
            image_metadata["Samples per pixel"] = \
                view.samples_per_pixel
            image_metadata["Number of derived levels"] = \
                self.num_derived_levels(img)

            for resolution in range(self.num_derived_levels(img)):
                dim_ranges = self.dimension_ranges(img, resolution)
                level_size_x = self.get_size(dim_ranges[0])
                level_size_y = self.get_size(dim_ranges[1])
                image_metadata["Level sizes #%s" % resolution] = {
                    "X": level_size_x,
                    "Y": level_size_y
                }
                if resolution == 0:
                    self.size_x = level_size_x
                    self.size_y = level_size_y
        elif image_type == "LABELIMAGE":
            self.label_x = self.get_size(view.dimension_ranges(0)[0]) + 1
            self.label_y = self.get_size(view.dimension_ranges(0)[1]) + 1
        elif image_type == "MACROIMAGE":
            self.macro_x = self.get_size(view.dimension_ranges(0)[0]) + 1
            self.macro_y = self.get_size(view.dimension_ranges(0)[1]) + 1
        return image_metadata

    def acquisition_datetime(self):
        pe_in = self.pixel_engine["in"]
        if self.sdk_v1:
            timestamp = str(pe_in.DICOM_ACQUISITION_DATETIME)
        else:
            timestamp = pe_in.acquisition_datetime
        return datetime.strptime(timestamp, "%Y%m%d%H%M%S.%f")

    def barcode(self):
        pe_in = self.pixel_engine["in"]
        if self.sdk_v1:
            return pe_in.BARCODE
        else:
            return pe_in.barcode

    def data_envelopes(self, image, resolution):
        pe_in = self.pixel_engine["in"]
        if self.sdk_v1:
            return pe_in.SourceView().dataEnvelopes(resolution)
        else:
            return image.source_view.data_envelopes(resolution)

    def derivation_description(self):
        pe_in = self.pixel_engine["in"]
        if self.sdk_v1:
            return pe_in.DICOM_DERIVATION_DESCRIPTION
        else:
            return pe_in.derivation_description

    def dimension_ranges(self, image, resolution):
        pe_in = self.pixel_engine["in"]
        if self.sdk_v1:
            return pe_in.SourceView().dimensionRanges(resolution)
        else:
            return image.source_view.dimension_ranges(resolution)

    def image_data(self, image):
        if self.sdk_v1:
            return image.IMAGE_DATA
        else:
            return image.image_data

    def image_type(self, image_no):
        pe_in = self.pixel_engine["in"]
        if self.sdk_v1:
            return pe_in[image_no].IMAGE_TYPE
        else:
            return pe_in[image_no].image_type

    def num_derived_levels(self, image):
        pe_in = self.pixel_engine["in"]
        if self.sdk_v1:
            return pe_in.numLevels()
        else:
            return image.source_view.num_derived_levels

    def num_images(self):
        pe_in = self.pixel_engine["in"]
        if self.sdk_v1:
            return pe_in.numImages()
        else:
            return pe_in.num_images

    def wait_any(self, regions):
        if self.sdk_v1:
            return self.pixel_engine.waitAny(regions)
        else:
            return self.pixel_engine.wait_any(regions)

    def write_metadata_json(self, metadata_file):
        '''write metadata to a JSON file'''

        with open(metadata_file, "w", encoding="utf-8") as f:
            metadata = self.get_metadata()

            for image in range(self.num_images()):
                image_metadata = self.get_image_metadata(image)
                metadata["Image #" + str(image)] = image_metadata

            json.dump(metadata, f)

    def write_metadata_xml(self, metadata_file):
        ome_timestamp = self.acquisition_datetime()

        xml_values = {
            'image': {
                'name': self.barcode(),
                'acquisitionDate': ome_timestamp.isoformat(),
                'description': self.derivation_description(),
                'pixels': {
                    'sizeX': int(self.size_x),
                    'sizeY': int(self.size_y),
                    'physicalSizeX': self.pixel_size_x,
                    'physicalSizeY': self.pixel_size_y
                }
            },
            'label': {
                'pixels': {
                    'sizeX': int(self.label_x),
                    'sizeY': int(self.label_y)
                }
            },
            'macro': {
                'pixels': {
                    'sizeX': int(self.macro_x),
                    'sizeY': int(self.macro_y)
                }
            }
        }
        loader = PackageLoader()
        template = loader.import_("isyntax2raw.resources.ome_template")
        xml = template(xml_values).render()
        with open(metadata_file, "w", encoding="utf-8") as omexml:
            omexml.write(xml)

    def write_metadata(self):
        os.makedirs(os.path.join(self.slide_directory, "OME"), exist_ok=True)

        metadata_file = os.path.join(
            self.slide_directory, "OME", "METADATA.json"
        )
        self.write_metadata_json(metadata_file)

        metadata_file = os.path.join(
            self.slide_directory, "OME", "METADATA.ome.xml"
        )
        self.write_metadata_xml(metadata_file)

    def get_size(self, dim_range):
        '''calculate the length in pixels of a dimension'''
        v = (dim_range[2] - dim_range[0]) / dim_range[1]
        if not v.is_integer():
            # isyntax infrastructure should ensure this always divides
            # evenly
            raise ValueError(
                '(%d - %d) / %d results in remainder!' % (
                    dim_range[2], dim_range[0], dim_range[1]
                )
            )
        return v

    def write_label_image(self):
        '''write the label image (if present) as a JPEG file'''
        self.write_image_type("LABELIMAGE", 1)

    def write_macro_image(self):
        '''write the macro image (if present) as a JPEG file'''
        self.write_image_type("MACROIMAGE", 2)

    def find_image_type(self, image_type):
        '''look up a given image type in the pixel engine'''
        pe_in = self.pixel_engine["in"]
        for index in range(self.num_images()):
            if image_type == self.image_type(index):
                return pe_in[index]
        return None

    def write_image_type(self, image_type, series):
        '''write an image of the specified type'''
        image = self.find_image_type(image_type)
        if image is not None:
            pixels = self.image_data(image)

            # pixels are JPEG compressed, need to decompress first
            img = Image.open(BytesIO(pixels))
            width = img.width
            height = img.height

            self.create_tile_directory(series, 0, width, height)
            tile = self.zarr_group["%d/0" % series]
            tile.attrs['image type'] = image_type
            for channel in range(0, 3):
                band = np.array(img.getdata(band=channel))
                band.shape = (height, width)
                tile[0, 0, channel] = band

            log.info("wrote %s image" % image_type)

    def create_tile_directory(self, series, resolution, width, height):
        dimension_separator = '/'
        if not self.nested:
            dimension_separator = '.'
        self.zarr_store = FSStore(
            self.slide_directory,
            dimension_separator=dimension_separator,
            normalize_keys=True,
            auto_mkdir=True
        )
        self.zarr_group = zarr.group(store=self.zarr_store)
        self.zarr_group.attrs['bioformats2raw.layout'] = LAYOUT_VERSION

        # important to explicitly set the chunk size to 1 for non-XY dims
        # setting to None may cause all planes to be chunked together
        # ordering is TZCYX and hard-coded since Z and T are not present
        self.zarr_group.create_dataset(
            "%s/%s" % (str(series), str(resolution)),
            shape=(1, 1, 3, height, width),
            chunks=(1, 1, 1, self.tile_height, self.tile_width), dtype='B'
        )

    def make_planar(self, pixels, tile_width, tile_height):
        r = pixels[0::3]
        g = pixels[1::3]
        b = pixels[2::3]
        for v in (r, g, b):
            v.shape = (tile_height, tile_width)
        return np.array([r, g, b])

    def write_pyramid(self):
        '''write the slide's pyramid as a set of tiles'''
        pe_in = self.pixel_engine["in"]
        image = self.find_image_type("WSI")

        scanned_areas = self.data_envelopes(image, 0)
        if scanned_areas is None:
            raise RuntimeError("No valid data envelopes")

        if self.resolutions is None:
            resolutions = range(self.num_derived_levels(image))
        else:
            resolutions = range(self.resolutions)

        def write_tile(
            pixels, resolution, x_start, y_start, tile_width, tile_height,
        ):
            x_end = x_start + tile_width
            y_end = y_start + tile_height
            try:
                # Zarr has a single n-dimensional array representation on
                # disk (not interleaved RGB)
                pixels = self.make_planar(pixels, tile_width, tile_height)
                z = self.zarr_group["0/%d" % resolution]
                z[0, 0, :, y_start:y_end, x_start:x_end] = pixels
            except Exception:
                log.error(
                    "Failed to write tile [:, %d:%d, %d:%d]" % (
                        x_start, x_end, y_start, y_end
                    ), exc_info=True
                )

        for resolution in resolutions:
            # assemble data envelopes (== scanned areas) to extract for
            # this level
            dim_ranges = self.dimension_ranges(image, resolution)
            log.info("dimension ranges = %s" % dim_ranges)
            resolution_x_size = self.get_size(dim_ranges[0])
            resolution_y_size = self.get_size(dim_ranges[1])
            scale_x = dim_ranges[0][1]
            scale_y = dim_ranges[1][1]

            x_tiles = math.ceil(resolution_x_size / self.tile_width)
            y_tiles = math.ceil(resolution_y_size / self.tile_height)

            log.info("# of X (%d) tiles = %d" % (self.tile_width, x_tiles))
            log.info("# of Y (%d) tiles = %d" % (self.tile_height, y_tiles))

            # create one tile directory per resolution level if required
            tile_directory = self.create_tile_directory(
                0, resolution, resolution_x_size, resolution_y_size
            )

            patches, patch_ids = self.create_patch_list(
                dim_ranges, [x_tiles, y_tiles],
                [self.tile_width, self.tile_height],
                tile_directory
            )
            envelopes = self.data_envelopes(image, resolution)
            jobs = []
            with MaxQueuePool(ThreadPoolExecutor, self.max_workers) as pool:
                for i in range(0, len(patches), self.batch_size):
                    # requestRegions(
                    #    self: pixelengine.PixelEngine.View,
                    #    region: List[List[int]],
                    #    dataEnvelopes: pixelengine.PixelEngine.DataEnvelopes,
                    #    enableAsyncRendering: bool=True,
                    #    backgroundColor: List[int]=[0, 0, 0],
                    #    bufferType:
                    #      pixelengine.PixelEngine.BufferType=BufferType.RGB
                    # ) -> list
                    if self.sdk_v1:
                        request_regions = pe_in.SourceView().requestRegions
                    else:
                        request_regions = image.source_view.request_regions
                    regions = request_regions(
                        patches[i:i + self.batch_size], envelopes, True,
                        [self.fill_color] * 3
                    )
                    while regions:
                        regions_ready = self.wait_any(regions)

                        for region_index, region in enumerate(regions_ready):
                            view_range = region.range
                            log.debug(
                                "processing tile %s (%s regions ready; "
                                "%s regions left; %s jobs)" % (
                                    view_range, len(regions_ready),
                                    len(regions), len(jobs)
                                )
                            )
                            x_start, x_end, y_start, y_end, level = view_range
                            width = 1 + (x_end - x_start) / scale_x
                            # isyntax infrastructure should ensure this always
                            # divides evenly
                            if not width.is_integer():
                                raise ValueError(
                                    '(1 + (%d - %d) / %d results in '
                                    'remainder!' % (
                                        x_end, x_start, scale_x
                                    )
                                )
                            width = int(width)
                            height = 1 + (y_end - y_start) / scale_y
                            # isyntax infrastructure should ensure this always
                            # divides evenly
                            if not height.is_integer():
                                raise ValueError(
                                    '(1 + (%d - %d) / %d results in '
                                    'remainder!' % (
                                        y_end, y_start, scale_y
                                    )
                                )
                            height = int(height)
                            pixel_buffer_size = width * height * 3
                            pixels = np.empty(pixel_buffer_size, dtype='B')
                            patch_id = patch_ids.pop(regions.index(region))
                            x_start, y_start = patch_id
                            x_start *= self.tile_width
                            y_start *= self.tile_height

                            region.get(pixels)
                            regions.remove(region)

                            jobs.append(pool.submit(
                                write_tile, pixels, resolution,
                                x_start, y_start, width, height
                            ))
            wait(jobs, return_when=ALL_COMPLETED)

    def create_patch_list(
        self, dim_ranges, tiles, tile_size, tile_directory
    ):
        resolution_x_end = dim_ranges[0][2]
        resolution_y_end = dim_ranges[1][2]
        origin_x = dim_ranges[0][0]
        origin_y = dim_ranges[1][0]
        tiles_x, tiles_y = tiles

        patches = []
        patch_ids = []
        scale_x = dim_ranges[0][1]
        scale_y = dim_ranges[1][1]
        # We'll use the X scale to calculate our level.  If the X and Y scales
        # are not eqivalent or not a power of two this will not work but that
        # seems *highly* unlikely
        level = math.log2(scale_x)
        if scale_x != scale_y or not level.is_integer():
            raise ValueError(
                "scale_x=%d scale_y=%d do not match isyntax format "
                "assumptions!" % (
                    scale_x, scale_y
                )
            )
        level = int(level)
        tile_size_x = tile_size[0] * scale_x
        tile_size_y = tile_size[1] * scale_y
        for y in range(tiles_y):
            y_start = origin_y + (y * tile_size_y)
            # Subtracting "scale_y" here makes no sense but it works and
            # reflects the isyntax SDK examples
            y_end = min(
                (y_start + tile_size_y) - scale_y, resolution_y_end - scale_y
            )
            for x in range(tiles_x):
                x_start = origin_x + (x * tile_size_x)
                # Subtracting "scale_x" here makes no sense but it works and
                # reflects the isyntax SDK examples
                x_end = min(
                    (x_start + tile_size_x) - scale_x,
                    resolution_x_end - scale_x
                )
                patch = [x_start, x_end, y_start, y_end, level]
                patches.append(patch)
                # Associating spatial information (tile X and Y offset) in
                # order to identify the patches returned asynchronously
                patch_ids.append((x, y))
        return patches, patch_ids
