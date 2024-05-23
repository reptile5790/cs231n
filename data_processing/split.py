# import cv2
# import imageio
# import math
import os
# import pandas as pd
# import rasterio
from PIL import Image
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import numpy as np
# import skimage as ski
# from rasterio import mask
# from skimage.exposure import equalize_adapthist, adjust_gamma, rescale_intensity


def save_image_tiles(input_directory, output_directory, tile_size=512):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over each image in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('tif'):
            input_image_path = os.path.join(input_directory, filename)
            # Load the image using PIL
            image = Image.open(input_image_path)
            width, height = image.size

            print(f'Standardizing Image: {filename}')

            slice_count = 0

            # Determine the number of rows and columns needed to achieve the desired size
            rows = height // tile_size
            cols = width // tile_size

            # Iterate over the split grid
            for row in range(rows):
                for col in range(cols):
                    # Calculate the slice boundaries
                    x_start = col * tile_size
                    y_start = row * tile_size
                    x_end = x_start + tile_size
                    y_end = y_start + tile_size

                    # Extract the slice from the image
                    slice_image = image.crop((x_start, y_start, x_end, y_end))
                    # Create a standardized base filename by removing the extension
                    base_filename = os.path.splitext(filename)[0]
                    slice_name = os.path.join(output_directory, f"{base_filename}_{slice_count:04d}.png")

                    # Save the slice as an image file
                    slice_image.save(slice_name)

                    slice_count += 1

            print(f'     {slice_count} tiles created and saved for {filename} in {output_directory}.\n')
    print(f'     Image Tiling Complete.\n\n')


def save_mask_tiles(input_directory, output_directory, tile_size=512):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over each image in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('tif'):
            input_image_path = os.path.join(input_directory, filename)
            # Load the image using PIL
            image = Image.open(input_image_path)
            width, height = image.size

            # Remove the final row and column of pixels
            if width > 1 and height > 1:
                image = image.crop((0, 0, width - 1, height - 1))
                width, height = image.size

            print(f'Standardizing Image: {filename}')

            slice_count = 0

            if width % tile_size != 0 and height % tile_size != 0:
                raise RuntimeError("Image is not compatible with tile size")

            # Determine the number of rows and columns needed to achieve the desired size
            rows = height // tile_size
            cols = width // tile_size

            # Iterate over the split grid
            for row in range(rows):
                for col in range(cols):
                    # Calculate the slice boundaries
                    x_start = col * tile_size
                    y_start = row * tile_size
                    x_end = x_start + tile_size
                    y_end = y_start + tile_size

                    # Extract the slice from the image
                    slice_image = image.crop((x_start, y_start, x_end, y_end))
                    # Create a standardized base filename by removing the extension
                    base_filename = os.path.splitext(filename)[0]
                    slice_name = os.path.join(output_directory, f"{base_filename}_{slice_count:04d}.png")

                    # Save the slice as an image file
                    slice_image.save(slice_name)

                    slice_count += 1

            print(f'     {slice_count} tiles created and saved for {filename} in {output_directory}.\n')
    print(f'     Mask Tiling Complete.\n\n')
