import nibabel as nib
import numpy as np

def load_nifti_pair(fixed_path, moving_path):
    fixed = nib.load(fixed_path).get_fdata()
    moving = nib.load(moving_path).get_fdata()
    fixed /= fixed.max()
    moving /= moving.max()
    return fixed, moving
