# import cv2
# import imageio
# import math
import os
import numpy as np
# import pandas as pd
# import rasterio
from PIL import Image
import time
from tqdm import tqdm
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import numpy as np
# import skimage as ski
# from rasterio import mask
# from skimage.exposure import equalize_adapthist, adjust_gamma, rescale_intensity

Image.MAX_IMAGE_PIXELS = 268468226  # For example

def save_image_tiles(input_directory, output_directory, tile_size=512):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over each image in the input directory with tqdm progress bar
    for filename in tqdm(os.listdir(input_directory), desc="Processing Images"):
        if filename.lower().endswith('tif'):
            start_time = time.time()
            input_image_path = os.path.join(input_directory, filename)
            # Load the image using PIL
            image = Image.open(input_image_path)
            width, height = image.size

            print(f'Standardizing Image: {filename}')

            slice_count = 0

            # Determine the number of rows and columns needed to achieve the desired size
            rows = height // tile_size
            cols = width // tile_size

            total_tiles = rows * cols

            # Single progress bar for both loops
            with tqdm(total=total_tiles, desc=f"Tiling {filename}") as pbar:
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
                        pbar.update(1)  # Update the progress bar for each tile

            end_time = time.time()
            print(f'     {slice_count} tiles created and saved for {filename} in {output_directory}.\n')
            print(f'     Elapsed Time for image = {end_time - start_time:.2f} seconds\n')
    print(f'     Image Tiling Complete.\n\n')

#
# def save_mask_tiles(input_directory, output_directory, tile_size=512, mask_values=None):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     # Iterate over each image in the input directory
#     for filename in os.listdir(input_directory):
#         if filename.lower().endswith('tif'):
#             input_image_path = os.path.join(input_directory, filename)
#             # Load the image using PIL
#             image = Image.open(input_image_path)
#             width, height = image.size
#
#             # Remove the final row and column of pixels
#             if width > 1 and height > 1:
#                 image = image.crop((0, 0, width - 1, height - 1))
#                 width, height = image.size
#
#             print(f'Standardizing Image: {filename}')
#
#             slice_count = 0
#
#             if width % tile_size != 0 and height % tile_size != 0:
#                 raise RuntimeError("Image is not compatible with tile size")
#
#             # Determine the number of rows and columns needed to achieve the desired size
#             rows = height // tile_size
#             cols = width // tile_size
#
#             # Iterate over the split grid
#             for row in range(rows):
#                 for col in range(cols):
#                     # Calculate the slice boundaries
#                     x_start = col * tile_size
#                     y_start = row * tile_size
#                     x_end = x_start + tile_size
#                     y_end = y_start + tile_size
#
#                     # Extract the slice from the image
#                     slice_image = image.crop((x_start, y_start, x_end, y_end))
#                     # Create a standardized base filename by removing the extension
#                     base_filename = os.path.splitext(filename)[0]
#                     slice_name = os.path.join(output_directory, f"{base_filename}_{slice_count:04d}.png")
#
#                     # Save the slice as an image file
#                     slice_image.save(slice_name)
#
#                     slice_count += 1
#
#             print(f'     {slice_count} tiles created and saved for {filename} in {output_directory}.\n')
#     print(f'     Mask Tiling Complete.\n\n')
#


def save_mask_tiles(input_directory, output_directory, tile_size=512, allowed_values=None):
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

            if width % tile_size != 0 or height % tile_size != 0:
                raise RuntimeError("Image is not compatible with tile size")

            # Determine the number of rows and columns needed to achieve the desired size
            rows = height // tile_size
            cols = width // tile_size

            # Convert the image to a numpy array
            image_array = np.array(image)

            # Iterate over the split grid
            for row in range(rows):
                for col in range(cols):
                    # Calculate the slice boundaries
                    x_start = col * tile_size
                    y_start = row * tile_size
                    x_end = x_start + tile_size
                    y_end = y_start + tile_size

                    # Extract the slice from the image array
                    slice_array = image_array[y_start:y_end, x_start:x_end]

                    # If allowed_values is provided, set every pixel value to 0 unless it is in the allowed values list
                    if allowed_values is not None:
                        slice_array = np.where(np.isin(slice_array, allowed_values), slice_array, 0)

                    # Convert the slice array back to an image
                    slice_image = Image.fromarray(slice_array)

                    # Create a standardized base filename by removing the extension
                    base_filename = os.path.splitext(filename)[0]
                    slice_name = os.path.join(output_directory, f"{base_filename}_{slice_count:04d}.png")

                    # Save the slice as an image file
                    slice_image.save(slice_name)

                    slice_count += 1

            print(f'     {slice_count} tiles created and saved for {filename} in {output_directory}.\n')
    print(f'     Mask Tiling Complete.\n\n')


def save_mask_numpy_tiles(input_directory, output_directory, tile_size=512, allowed_values=None):
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

            if width % tile_size != 0 or height % tile_size != 0:
                raise RuntimeError("Image is not compatible with tile size")

            # Determine the number of rows and columns needed to achieve the desired size
            rows = height // tile_size
            cols = width // tile_size

            # Convert the image to a numpy array
            image_array = np.array(image)

            # Iterate over the split grid
            for row in range(rows):
                for col in range(cols):
                    # Calculate the slice boundaries
                    x_start = col * tile_size
                    y_start = row * tile_size
                    x_end = x_start + tile_size
                    y_end = y_start + tile_size

                    # Extract the slice from the image array
                    slice_array = image_array[y_start:y_end, x_start:x_end]

                    # If allowed_values is provided, set every pixel value to 0 unless it is in the allowed values list
                    if allowed_values is not None:
                        slice_array = np.where(np.isin(slice_array, allowed_values), slice_array, 0)

                    # Create a standardized base filename by removing the extension
                    base_filename = os.path.splitext(filename)[0]
                    slice_name = os.path.join(output_directory, f"{base_filename}_{slice_count:04d}.npy")

                    # Save the slice as a numpy array file
                    np.save(slice_name, slice_array)

                    slice_count += 1

            print(f'     {slice_count} tiles created and saved for {filename} in {output_directory}.\n')
    print(f'     Mask Tiling Complete.\n\n')