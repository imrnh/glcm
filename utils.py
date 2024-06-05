import numpy as np
import math

def write_glcm_step(glcm_image, index, args):
    """
    Write the full array into corresponding file.

    Args:
        glcm_image: the actual glcm_image at i'th step
        index: i value
    """


    with open(f"sims/{index}.txt", "w") as f:

        if args != None:
            f.write(f"i: {args[0]}\nj: {args[1]}\nshifted_row: {args[2]}\nshifted_col: {args[3]}\n")
            f.write(f"Intensity i value (GLSM row idx): {args[4]}\n")
            f.write(f"Intensity j value (GLSM col idx): {args[5]}\n")
            f.write("\n\n\n\n")


        for row in glcm_image:
            row_string = '\t\t'.join(map(str, row))
            f.write(row_string + '\n\n')



def get_entropy(n_glcm):
    entropy = 0
    epsilon = 0.00001
    for row in n_glcm:
        for ng_val in row:
            entropy += -(ng_val * math.log2(ng_val + epsilon)) # To prevent log(0)
    
    return round(entropy, 2)


def get_contrast(n_glcm):
    contrast = 0
    for r_idx, row in enumerate(n_glcm):
        for c_idx, ng_val in enumerate(row):
            contrast += ((r_idx - c_idx) ** 2) * ng_val 
            
    return round(contrast, 2)


def get_homogeneity(n_glcm):
    hgty = 0
    for r_idx, row in enumerate(n_glcm):
        for c_idx, ng_val in enumerate(row):
            hgty += ng_val / (1 + abs(r_idx - c_idx))
            
    return round(hgty, 2)

def inverse_diff_moment(n_glcm):
    idm = 0
    for r_idx, row in enumerate(n_glcm):
        for c_idx, ng_val in enumerate(row):
            idm += ng_val / (1 + ((r_idx - c_idx)**2))
            
    return round(idm, 2)

def get_angular_2nd_mom(n_glcm):
    return np.sum(n_glcm**2)


def get_correlation(n_glcm):
    i_indices, j_indices = np.indices(n_glcm.shape)
    mu_i = np.sum(i_indices * n_glcm)
    mu_j = np.sum(j_indices * n_glcm)
    sigma_i = np.sqrt(np.sum((i_indices - mu_i)**2 * n_glcm))
    sigma_j = np.sqrt(np.sum((j_indices - mu_j)**2 * n_glcm))
    correlation = np.sum((i_indices - mu_i) * (j_indices - mu_j) * n_glcm) / (sigma_i * sigma_j)

    return correlation


def get_sum_variance(n_glcm, max_intensity):
    px_plus_y = [np.sum(n_glcm[k - l, l]) for k in range(2 * max_intensity) for l in range(max_intensity) if 0 <= k - l < max_intensity]
    sum_mean = np.sum([i * px_plus_y[i] for i in range(len(px_plus_y))])
    sum_variance = np.sum([(i - sum_mean) ** 2 * px_plus_y[i] for i in range(len(px_plus_y))])

    return sum_variance

def get_info_measure_correlation(n_glcm):
    epsilon = 1e-10
    px = np.sum(n_glcm, axis=1)
    py = np.sum(n_glcm, axis=0)
    H_xy = -np.sum(n_glcm * np.log(n_glcm + epsilon))
    H_x = -np.sum(px * np.log(px + epsilon))
    H_y = -np.sum(py * np.log(py + epsilon))
    H_xy1 = -np.sum(n_glcm * np.log(np.outer(px, py) + epsilon))
    IMC1 = (H_xy - H_xy1) / max(H_x, H_y)
    IMC2 = np.sqrt(max(0, 1 - np.exp(-2 * (H_xy - H_xy1))))

    return IMC1, IMC2