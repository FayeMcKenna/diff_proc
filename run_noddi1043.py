'''
============================================================
Neurite Orientation and Distribution dmipy
============================================================
'''

#import modules
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy, acquisition_scheme_from_bvalues
import numpy as np
from dipy.reconst.ivim import IvimModel
import nibabel as nib
from dipy.segment.mask import applymask, bounding_box, crop
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

"""
 Load and set up Watson Noddi model

"""
from dmipy.signal_models import cylinder_models, gaussian_models

ball = gaussian_models.G1Ball()
stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()

from dmipy.distributions.distribute_models import SD1WatsonDistributed

watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
watson_dispersed_bundle.parameter_names

watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par',
                                               'partial_volume_0')
watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

"""
Put model together 
"""
from dmipy.core.modeling_framework import MultiCompartmentModel

NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
NODDI_mod.parameter_names
print(NODDI_mod.parameter_names)
NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

"""
Load data and set mask 
"""
# load directory names
input_directory = "/gpfs/home/fm1545/Faye/Diff_models/toprocess/1043"
output_directory = "/gpfs/home/fm1545/Faye/Diff_models/toprocess/1043"

# filenames -- mask has all NaN removed to work with function
fivim = input_directory + "/noddidata.nii.gz"
fmask = input_directory + "/mask.nii.gz"
fbval = input_directory + "/noddibvals.txt"
fbvec = input_directory + "/noddibvecs.txt"

# load bvals, bvecs and gradient files
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)
bvalues_SI = bvals * 1e6  # now given in SI units as s/m^2

# The delta and Delta times we know from the HCP documentation in seconds
delta = 0.0106
Delta = 0.0431

acq_scheme = acquisition_scheme_from_bvalues(bvalues_SI, bvecs, delta, Delta)
acq_scheme.print_acquisition_info
acq_scheme.bvalues;  # bvalues in s/m^2
acq_scheme.gradient_directions;  # gradient directions on the unit sphere
acq_scheme.gradient_strengths;  # the gradient strength in T/m
acq_scheme.qvalues;  # describes the diffusion sensitization in 1/m
acq_scheme.tau;  # diffusion time as Delta - delta / 3. in seconds
acq_scheme.shell_indices;  # index assigned to each shell. 0 is assigned to b0 measurements

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
Run Model
"""

NODDI_fit_hcp = NODDI_mod.fit(acq_scheme, data, mask=mask)

fitted_parameters = NODDI_fit_hcp.fitted_parameters

save_nifti(output_directory + '/NoddiODI.nii.gz',fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi'], affine)
save_nifti(output_directory + '/Noddiwm.nii.gz', fitted_parameters['partial_volume_1'], affine)
save_nifti(output_directory + '/Noddicsf.nii.gz', fitted_parameters['partial_volume_0'], affine)
save_nifti(output_directory + '/Noddistick.nii.gz',fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'], affine)

# get total Stick signal contribution
vf_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
# get total Zeppelin signal contribution
vf_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) * fitted_parameters['partial_volume_1'])

save_nifti(output_directory + '/Noddivfintra.nii.gz', fitted_parameters['Vf_intra'], affine)
save_nifti(output_directory + '/Noddivfextra.nii.gz', fitted_parameters['Vf_extra'], affine)
save_nifti(output_directory + '/Noddivfintra2.nii.gz', Vf_intra, affine)
save_nifti(output_directory + '/Noddivfextra2.nii.gz', Vf_extra, affine)


