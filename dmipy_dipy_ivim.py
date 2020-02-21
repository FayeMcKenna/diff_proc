'''
============================================================
Intravoxel incoherent motion dmipy
============================================================
'''

#import modules
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy, acquisition_scheme_from_bvalues
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from dipy.reconst.ivim import IvimModel
import nibabel as nib
from dipy.segment.mask import applymask, bounding_box, crop
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

"""
Load data and set mask 
"""
# load directory names
input_directory = "/Users/fayemckenna/Desktop/ivimtest2"
#working_directory = "data/working"
output_directory = "/Users/fayemckenna/Desktop/ivimtest2"

# filenames -- mask has all NaN removed to work with function
fivim = input_directory + "/ivimdata.nii.gz"
fmask = input_directory + "/mask.nii.gz"
fbval = input_directory + "/ivimbval.txt"
fbvec = input_directory + "/ivimbvec.txt"

# load bvals, bvecs and gradient files
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs, b0_threshold=1e6)

# Load the data with  mask
img = nib.load(fivim)
img_data = img.get_fdata()
# define affine space
data1, affine = load_nifti(fivim)
# Load the mask
mask = nib.load(fmask)
mask_data = mask.get_fdata()

# Apply the mask to the volume
mask_boolean = mask_data > 0.01
mins, maxs = bounding_box(mask_boolean)
mask_boolean = crop(mask_boolean, mins, maxs)
cropped_volume = crop(img_data, mins, maxs)
data = applymask(cropped_volume, mask_boolean)

"""
dmipy processing
"""
# look at B shells
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy, acquisition_scheme_from_bvalues
scheme_ivim = gtab_dipy2dmipy(gtab, b0_threshold=1e6, min_b_shell_distance=1e6)
scheme_ivim.print_acquisition_info

#select only slice/voxel for processing
#data_slice = data[90: 155, 90: 170, 33, :]
#test_voxel = data_slice[0, 0]

#fit model
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models.gaussian_models import G1Ball
ivim_mod = MultiCompartmentModel([G1Ball(), G1Ball()])
ivim_mod.set_fixed_parameter(
    'G1Ball_2_lambda_iso', 7e-9)  # Following Gurney-Champion 2016
ivim_mod.set_parameter_optimization_bounds(
    'G1Ball_1_lambda_iso', [.5e-9, 6e-9])  # Following Gurney-Champion 2016
ivim_fit_Dfixed = ivim_mod.fit(
    acquisition_scheme=scheme_ivim,
    data=data)

"""
dipy processing as reference
"""
from dipy.reconst.ivim import IvimModel
ivimmodel = IvimModel(gtab)
ivim_fit_dipy = ivimmodel.fit(data)

"""
Parameter map for dmipy model
"""
from dmipy.custom_optimizers.intra_voxel_incoherent_motion import ivim_Dstar_fixed
from time import time
ivim_fit_dmipy_fixed = ivim_Dstar_fixed(scheme_ivim, data)
dipy_start = time()
ivim_fit_dipy = ivimmodel.fit(data)
print('Dipy computation time: {} s'.format(time() - dipy_start))

#save
save_nifti(output_directory + '/IvimDfixperf.nii.gz', ivim_fit_dmipy_fixed.fitted_parameters['partial_volume_1'],affine)
save_nifti(output_directory + '/IvimDfixdiff.nii.gz', ivim_fit_dmipy_fixed.fitted_parameters['G1Ball_1_lambda_iso'],affine)
save_nifti(output_directory + '/Ivimdipyperf.nii.gz', ivim_fit_dipy.perfusion_fraction,affine)
save_nifti(output_directory + '/Ivimdipydiff.nii.gz', ivim_fit_dipy.D,affine)