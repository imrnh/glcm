import os
import numpy as np
from glcm import make_glcm


if __name__ == "__main__":
    os.makedirs("sims", exist_ok=True)

    image = np.array(

        # your image as 2d array.

    )

    image_rounded = np.rint(image).astype(int)
    displacement = [0, -1]
    glcm = make_glcm(image_rounded, displacement, levels=9)
    normalized_glcm = glcm / sum(sum(glcm))

    print(glcm)
    print(normalized_glcm)