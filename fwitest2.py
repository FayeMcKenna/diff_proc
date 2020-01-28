"""
FWI on Lazar Lab RO1 data 2020
adapted from DIPY code
Faye McKenna
"""
# Load the necessary python libraries
import os, sys
import numpy as np
import nibabel as nib
from dipy.segment.mask import applymask, bounding_box, crop
import dipy.reconst.fwdti as fwdti
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

# load directory names
input_directory = "/Users/fm1545/Desktop/dipy_data/r01_multib"
#working_directory = "data/working"
output_directory = "/Users/fm1545/Desktop/dipy_data/r01_multib"

# filenames -- mask has all NaN removed to work with function
fdwi = input_directory + "/dwi1001.nii.gz"
fmask = input_directory + "/mask2.nii.gz"
fbval = input_directory + "/bvals.txt"
fbvec = input_directory + "/bvecs.txt"

# Load the data
img = nib.load(fdwi)
img_data = img.get_fdata()
# define affine space
data1, affine = load_nifti(fdwi)
# Load the mask
mask = nib.load(fmask)
mask_data = mask.get_fdata()

# load bvals, bvecs and gradient files
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

# Apply the mask to the volume
mask_boolean = mask_data > 0.01
mins, maxs = bounding_box(mask_boolean)
mask_boolean = crop(mask_boolean, mins, maxs)
cropped_volume = crop(img_data, mins, maxs)
data = applymask(cropped_volume, mask_boolean)

# Run the Dipy FWI calculation
fwdtimodel = fwdti.FreeWaterTensorModel(gtab)
fwdtifit = fwdtimodel.fit(data, mask=mask_boolean)

# define FW FA and MD variables
FA = fwdtifit.fa
MD = fwdtifit.md

# run the classical DTI model as reference
dtimodel = dti.TensorModel(gtab)
dtifit = dtimodel.fit(data, mask=mask_boolean)

#define classic FA and MD variables
dti_FA = dtifit.fa
dti_MD = dtifit.md

# define FW image variable
F = fwdtifit.f

#save variables as nifti images
save_nifti('WB_regMD.nii.gz',dti_MD,affine)
save_nifti('WB_regFA.nii.gz',dti_FA,affine)
save_nifti('WB_fwi.nii.gz',F,affine)
save_nifti('WB_fwiFA.nii.gz',FA,affine)
save_nifti('WB_fwiMD.nii.gz',MD,affine)


