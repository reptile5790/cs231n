import os
import numpy as np
from PIL import Image
import time
from tqdm.notebook import tqdm

Image.MAX_IMAGE_PIXELS = 268468226

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


def save_mask_tiles(input_directory, output_directory, tile_size=512, allowed_values=None):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over each image in the input directory with tqdm progress bar
    for filename in tqdm(os.listdir(input_directory), desc="Processing Images"):
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

                        # Extract the slice from the image array
                        slice_array = np.array(image.crop((x_start, y_start, x_end, y_end)))

                        # If allowed_values is provided, remap the pixel values
                        if allowed_values is not None:
                            remapped_slice = np.zeros_like(slice_array)
                            for original_value, new_value in allowed_values.items():
                                remapped_slice[slice_array == original_value] = new_value
                            slice_array = remapped_slice

                        # Create a standardized base filename by removing the extension
                        base_filename = os.path.splitext(filename)[0]
                        slice_path = os.path.join(output_directory, f"{base_filename}_{slice_count:04d}.png")

                        # Save slice as PNG
                        slice_image = Image.fromarray(slice_array.astype(np.uint8))
                        slice_image.save(slice_path)

                        slice_count += 1
                        pbar.update(1)  # Update the progress bar for each tile

            print(f'     {slice_count} tiles created and saved for {filename} in {output_directory}.\n')
    print(f'     Mask Tiling Complete.\n\n')


def save_mask_numpy_tiles(input_directory, output_directory, tile_size=512, allowed_values=None):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over each image in the input directory with tqdm progress bar
    for filename in tqdm(os.listdir(input_directory), desc="Processing Images"):
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

                        # Extract the slice from the image array
                        slice_array = np.array(image.crop((x_start, y_start, x_end, y_end)))

                        # If allowed_values is provided, remap the pixel values
                        if allowed_values is not None:
                            remapped_slice = np.zeros_like(slice_array)
                            for original_value, new_value in allowed_values.items():
                                remapped_slice[slice_array == original_value] = new_value
                            slice_array = remapped_slice

                        # Create a standardized base filename by removing the extension
                        base_filename = os.path.splitext(filename)[0]
                        slice_name = os.path.join(output_directory, f"{base_filename}_{slice_count:04d}.npy")

                        # Save the slice as a numpy array file
                        np.save(slice_name, slice_array)

                        slice_count += 1
                        pbar.update(1)  # Update the progress bar for each tile

            print(f'     {slice_count} tiles created and saved for {filename} in {output_directory}.\n')
    print(f'     Mask Tiling Complete.\n\n')