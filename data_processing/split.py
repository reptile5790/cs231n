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


    # Load the image using PIL
    image = Image.open(input_image_path)
    width, height = image.size

    print(f'Standardizing Image.')

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
            slice_name = os.path.join(output_directory, f"tile_{slice_count:04d}.png")

            # Save the slice as an image file
            slice_image.save(slice_name)

            slice_count += 1

    print(f'     {slice_count} tiles created and saved in {output_directory}.\n'
          f'     Standardization Complete.\n\n')


def save_mask_tiles(input_directory, output_directory, tile_size=512):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the image using PIL
    image = Image.open(input_image_path)
    width, height = image.size

    print(f'Standardizing Image.')

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
            slice_name = os.path.join(output_directory, f"tile_{slice_count:04d}.png")

            # Save the slice as an image file
            slice_image.save(slice_name)

            slice_count += 1

    print(f'     {slice_count} tiles created and saved in {output_directory}.\n'
          f'     Standardization Complete.\n\n')


# Example usage:
input_image_path = 'D:/cs231n_data/2020/july/L15-0469E-1284N.tif'
output_dir = 'D:/cs231n_data/test'

# Call the function
save_image_tiles(input_image_path, output_directory=output_dir)
