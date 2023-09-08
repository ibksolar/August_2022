# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:17:44 2023

@author: i368o351
"""

import cv2
import numpy as np


from skimage import io, color, filters
import matplotlib.pyplot as plt

def auto_region_growing(image, similarity_threshold):
    # Create a mask of the same size as the input image, initialized to zeros
    mask = np.zeros_like(image)

    # Get image dimensions
    height, width = image.shape

    # Define connectivity (e.g., 8-connectivity for neighboring pixels)
    connectivity = 8

    # Loop through all pixels in the image
    for y in range(height):
        for x in range(width):
            # Check if the pixel is unprocessed and above the threshold
            if mask[y, x] == 0 and image[y, x] > similarity_threshold:
                # Create a new region label
                label = np.max(mask) + 1

                # Initialize a stack with the current pixel
                stack = [(x, y)]

                while stack:
                    px, py = stack.pop()

                    # Mark the pixel as part of the region
                    mask[py, px] = label

                    # Check neighboring pixels for similarity and add them to the stack
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = px + dx, py + dy

                            # Check if the neighboring pixel is within the image boundaries
                            if 0 <= nx < width and 0 <= ny < height:
                                # Check similarity criterion and unprocessed status
                                if (
                                    mask[ny, nx] == 0
                                    and image[ny, nx] > similarity_threshold
                                ):
                                    stack.append((nx, ny))

    return mask

# Read your image (grayscale)
image = cv2.imread('your_image.png', cv2.IMREAD_GRAYSCALE)

# Define a similarity threshold (adjust as needed)
similarity_threshold = 0.5

# Apply automatic region growing
segmented_regions = auto_region_growing(image, similarity_threshold)

# Display the segmented regions
cv2.imshow('Segmented Regions', segmented_regions)
cv2.waitKey(0)
cv2.destroyAllWindows()




### Custom Adaptive Threshold

# Load your image
image = io.imread('your_image.png')

# Convert the image to grayscale if it's in color
if image.shape[-1] == 3:
    image = color.rgb2gray(image)

# Define the size of the local neighborhood for thresholding
block_size = 35  # Adjust as needed

# Apply adaptive thresholding using a custom function
def custom_adaptive_threshold(image, block_size):
    thresholded_image = np.zeros_like(image)

    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            # Extract the local neighborhood
            neighborhood = image[row:row+block_size, col:col+block_size]

            # Calculate a custom threshold for the neighborhood (e.g., mean or median)
            threshold = np.mean(neighborhood)  # You can use other criteria here

            # Threshold the neighborhood
            neighborhood_thresholded = (neighborhood > threshold).astype(np.uint8)

            # Place the thresholded neighborhood back into the result
            thresholded_image[row:row+block_size, col:col+block_size] = neighborhood_thresholded

    return thresholded_image

# Apply the custom adaptive thresholding function
thresholded_image = custom_adaptive_threshold(image, block_size)

# Display the result
plt.imshow(thresholded_image, cmap='gray')
plt.axis('off')
plt.show()

