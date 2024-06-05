import math

def write_glcm_step(glcm_image, index, args):
    """
    Write the full array into corresponding file.

    Args:
        glcm_image: the actual glcm_image at i'th step
        index: i value
    """


    with open(f"sims/{index}.txt", "w") as f:
        f.write(f"i: {args[0]}\nj: {args[1]}\nshifted_row: {args[2]}\nshifted_col: {args[3]}\n")
        f.write(f"Intensity i value (GLSM row idx): {args[4]}\n")
        f.write(f"Intensity j value (GLSM col idx): {args[5]}\n")
        f.write("\n\n\n\n")


        for row in glcm_image:
            row_string = '\t\t'.join(map(str, row))
            f.write(row_string + '\n')



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
    mom = 0
    for r_idx, row in enumerate(n_glcm):
        for c_idx, ng_val in enumerate(row):
            mom += ng_val ** 2
    
    return round(mom, 2)