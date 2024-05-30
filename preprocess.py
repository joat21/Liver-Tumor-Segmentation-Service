import nibabel as nib
import numpy as np
from torchvision import transforms

def get_nii_slices(volume):
    volume_slices = []
    
    vol_nii = nib.load(volume).get_fdata()
    vol_nii = vol_nii.transpose(2, 1, 0)
    
    for vol in vol_nii:
        volume_slices.append(vol.astype('float32'))
                    
    return volume_slices
            
def normalize_nii(files):
    
    for i in range(len(files)):
        files[i][files[i] < 0] = 0
        files[i] = files[i] / np.max(files[i])
        
    return files
