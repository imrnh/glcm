import os
import numpy as np
from glcm import make_glcm
from utils import get_entropy, get_contrast, get_homogeneity, inverse_diff_moment, get_angular_2nd_mom
from store import image

if __name__ == "__main__":
    os.makedirs("sims", exist_ok=True)

    image_rounded = np.rint(image).astype(int)
    displacement = [0, -1]
    glcm = make_glcm(image_rounded, displacement, levels=9)
    normalized_glcm = glcm / sum(sum(glcm))

    print(f"GLCM: {glcm}")
    print(f"Normalized GLCM: {normalized_glcm}")

    print(f"Entropy: {get_entropy(normalized_glcm)}")
    print(f"Contrast: {get_contrast(normalized_glcm)}")
    print(f"Homogeneity: {get_homogeneity(normalized_glcm)}")
    print(f"Inverse Difference Moment: {inverse_diff_moment(normalized_glcm)}")
    print(f"Angular 2nd momentum: {get_angular_2nd_mom(normalized_glcm)}")