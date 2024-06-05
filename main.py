import os
import numpy as np
from glcm import make_glcm
from utils import get_entropy, get_contrast, get_homogeneity, inverse_diff_moment, get_angular_2nd_mom

if __name__ == "__main__":
    os.makedirs("sims", exist_ok=True)

    image = [
        [0, 3, 0, 8, 1, 6, 6, 2, 1, 1, 0, 2, 4],
        [0, 9, 2, 2, 5, 3, 2, 4, 5, 0, 2, 5, 1],
        [3, 6, 1, 0, 0, 5, 7, 0, 0, 6, 1, 0, 0],
        [5, 3, 7, 3, 0, 2, 5, 3, 2, 0, 7, 3, 6],
        [5, 5, 5, 0, 1, 3, 7, 4, 1, 5, 5, 5, 0],
        [5, 4, 7, 0, 5, 4, 2, 6, 5, 6, 7, 7, 5],
        [6, 3, 0, 2, 8, 6, 4, 3, 0, 5, 0, 3, 6],
        [7, 5, 1, 0, 4, 3, 3, 6, 4, 6, 1, 7, 5],
        [4, 3, 4, 2, 9, 6, 4, 3, 1, 7, 4, 3, 6],
        [1, 3, 3, 4, 2, 2, 2, 7, 4, 4, 3, 8, 7],
        [3, 7, 1, 8, 1, 6, 0, 2, 1, 7, 1, 2, 4],
        [7, 4, 1, 4, 3, 1, 2, 3, 5, 4, 1, 3, 7],
        [0, 1, 2, 9, 3, 7, 2, 6, 3, 4, 2, 7, 4]
    ]

    image_rounded = np.rint(np.array(image)).astype(int)
    displacement = [0, 2]
    glcm = make_glcm(image_rounded, displacement, levels=9)
    normalized_glcm = glcm / sum(sum(glcm))

    print(f"GLCM: {glcm}")
    print(f"Normalized GLCM: {normalized_glcm}")

    print(f"Entropy: {get_entropy(normalized_glcm)}")
    print(f"Contrast: {get_contrast(normalized_glcm)}")
    print(f"Homogeneity: {get_homogeneity(normalized_glcm)}")
    print(f"Inverse Difference Moment: {inverse_diff_moment(normalized_glcm)}")
    print(f"Angular 2nd momentum: {get_angular_2nd_mom(normalized_glcm)}")