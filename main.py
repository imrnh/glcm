import os
import numpy as np
from glcm import make_glcm
from utils import get_entropy, get_contrast, get_homogeneity, inverse_diff_moment, get_angular_2nd_mom

if __name__ == "__main__":
    os.makedirs("sims", exist_ok=True)

    image = np.array([
    [0, 9, 2, 8, 1, 3, 4, 2, 1, 3, 2, 2, 4],
    [7, 6, 1, 6, 2, 4, 6, 1, 4, 3, 1, 1.5, 3],
    [7, 5, 3, 7, 5, 6, 1, 5, 7, 5, 3, 6.5, 3],
    [2, 8, 6, 1, 6, 0, 2, 6, 6, 3, 6, 7.5, 5],
    [3, 6, 1, 6, 2, 5, 3, 1, 4, 1, 1, 1.5, 3],
    [5, 3, 7, 2, 4, 3, 7, 4, 4, 2, 7, 5.5, 1],
    [8, 0, 6, 5, 0, 4, 0, 5, 2, 4, 6, 6, 2],
    [1, 3, 3, 8, 0, 2, 0, 6, 0, 3, 3, 7, 4],
    [7, 7, 1, 6, 3, 7, 5, 5, 5, 7, 1, 6.5, 3],
    [2, 0, 6, 4, 3, 5, 7, 3, 5, 3, 6, 3.5, 7],
    [8, 8, 6, 7, 4, 1, 6, 1, 6, 6, 6, 1.5, 3],
    [7, 2, 3, 2, 8, 0, 6, 3, 0, 1, 3, 3, 6],
    [0, 9, 2, 2, 4, 2, 3, 0, 4, 1, 2, 0.5, 1]
])

    image_rounded = np.rint(np.ceil(np.array(image))).astype(int)
    displacement = [0, -1]
    glcm = make_glcm(image_rounded, displacement, levels=9)
    normalized_glcm = glcm / sum(sum(glcm))
    normalized_glcm = np.round(normalized_glcm, 2)

    print(f"GLCM: {glcm}")
    print(f"Normalized GLCM:\n {normalized_glcm}")

    print(f"Entropy: {get_entropy(normalized_glcm)}")
    print(f"Contrast: {get_contrast(normalized_glcm)}")
    print(f"Homogeneity: {get_homogeneity(normalized_glcm)}")
    print(f"Inverse Difference Moment: {inverse_diff_moment(normalized_glcm)}")
    print(f"Angular 2nd momentum: {get_angular_2nd_mom(normalized_glcm)}")