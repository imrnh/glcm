import numpy as np
from PIL import Image


def write_glcm_step(glcm_image, index):
    """
    Write the full array into corresponding file.

    Args:
        glcm_image: the actual glcm_image at i'th step
        index: i value
    """


    with open(f"sims/{index}.txt", "w") as f:
        for row in glcm_image:
            row_string = ' '.join(map(str, row))
            f.write(row_string + '\n')



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
    glcm = np.zeros((levels+1, levels+1))


    step_count = 0
    for row in range(rows):
        for col in range(cols):
            intensity_i = image[row, col]
            row_offset, col_offset = displacement

            neighbor_row = row + row_offset
            neighbor_col = col + col_offset
            if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                step_count += 1
                intensity_j = image[neighbor_row, neighbor_col]
                glcm[intensity_i, intensity_j] += 1
                write_glcm_step(glcm, step_count)

    return glcm

