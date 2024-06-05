import numpy as np
from utils import write_glcm_step


def make_glcm(image, displacement, levels):
    """
    Calculate the Gray Level Co-occurrence Matrix (GLCM) for the given image.

    Args:
        image (numpy arra): Input image
        displacement (list): The displacement vector given in the sheet [row_offset, col_offset]
        levels (int): Maximum pixel value of the image. 

    Returns:
        The GLCM Array
    """
    rows, cols = image.shape
    glcm = np.zeros((levels+1, levels+1), dtype=np.uint8)


    step_count = 0
    for row in range(rows):
        for col in range(cols):
            intensity_i = image[row, col]
            row_offset, col_offset = displacement

            neighbor_row = row + col_offset
            neighbor_col = col + row_offset
            if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                step_count += 1
                intensity_j = image[neighbor_row, neighbor_col]
                glcm[intensity_i, intensity_j] += 1
                write_glcm_step(glcm, step_count, [row, col, neighbor_row, neighbor_col, intensity_i, intensity_j])

    return glcm

