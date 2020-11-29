'''
============================================================
Intravoxel incoherent motion dmipy
============================================================
'''

#import modules
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core import modeling_framework
import numpy as np
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy, acquisition_scheme_from_bvalues
from dipy.reconst.ivim import IvimModel
import nibabel as nib
from dipy.segment.mask import applymask, bounding_box, crop
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dmipy.utils import utils

"""
Load data and set mask 
"""

# load directory names
input_directory = "/gpfs/home/fm1545/Faye/Diff_models/toprocess/1001"
output_directory = "/gpfs/home/fm1545/Faye/Diff_models/toprocess/1001"

# filenames -- mask has all NaN removed to work with function
fivim = input_directory + "/ivimdata.nii.gz"
fmask = input_directory + "/mask.nii.gz"
fbval = input_directory + "/ivimbvals.txt"
fbvec = input_directory + "/ivimbvecs.txt"

 # load bvals, bvecs and gradient files
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)
bvalues_SI = bvals * 1e6  # now given in SI units as s/m^2

# The delta and Delta times we know from the HCP documentation in seconds
delta = 0.0106
Delta = 0.0431

from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
scheme_ivim = acquisition_scheme_from_bvalues(bvalues_SI, bvecs, delta, Delta, b0_threshold=1e6, min_b_shell_distance=1e7)
scheme_ivim.print_acquisition_info
scheme_ivim.bvalues;  # bvalues in s/m^2
scheme_ivim.gradient_directions;  # gradient directions on the unit sphere
scheme_ivim.gradient_strengths;  # the gradient strength in T/m
scheme_ivim.qvalues;  # describes the diffusion sensitization in 1/m
scheme_ivim.tau;  # diffusion time as Delta - delta / 3. in seconds
scheme_ivim.shell_indices;  # index assigned to each shell. 0 is assigned to b0 measurements


# run the classical DTI model as reference
from dmipy.core.acquisition_scheme import gtab_dmipy2dipy
gtab = gtab_dmipy2dipy(scheme_ivim)
print(gtab.bvals)


# Load the data with  mask
img = nib.load(fivim)
img_data = img.get_fdata()
# define affine space
data, affine = load_nifti(fivim)
# Load the mask
#mask = nib.load(fmask)
#mask_data = mask.get_fdata()

# Apply the mask to the volume
#mask_boolean = mask_data > 0.01
#mins, maxs = bounding_box(mask_boolean)
#mask_boolean = crop(mask_boolean, mins, maxs)
#cropped_volume = crop(img_data, mins, maxs)
#data = applymask(cropped_volume, mask_boolean)

"""
dmipy processing
"""

# fit model
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models.gaussian_models import G1Ball
from dmipy.signal_models.cylinder_models import C2CylinderStejskalTannerApproximation

cyl = cylinder_models.C2CylinderStejskalTannerApproximation()
ivim_mod = MultiCompartmentModel([G1Ball(),G1Ball(),C2CylinderStejskalTannerApproximation()])
ivim_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)  # Following Gurney-Champion 2016
ivim_mod.set_fixed_parameter('G1Ball_2_lambda_iso', 10e-9)  # Following Gurney-Champion 2016
#ivim_mod.set_fixed_parameter('C2CylinderStejskalTannerApproximation_1_lambda_par', 1.7e-9),
#ivim_mod.set_fixed_parameter('C2CylinderStejskalTannerApproximation_1_diameter', 6e-6),
ivim_fit_Dfixed = ivim_mod.fit(acquisition_scheme=scheme_ivim,data=data)

from dmipy.custom_optimizers.intra_voxel_incoherent_motioncyl2 import ivim_Dstar_fixed
from time import time
ivim_fit_dmipy_fixed = ivim_Dstar_fixed(scheme_ivim, data)
fitted_params = ivim_fit_dmipy_fixed.fitted_parameters


save_nifti(output_directory + '/IvimDfixFWfrac_cyltest.nii.gz',ivim_fit_dmipy_fixed.fitted_parameters['partial_volume_0'], affine)
save_nifti(output_directory + '/IvimDfixperffrac_cyltest.nii.gz',ivim_fit_dmipy_fixed.fitted_parameters['partial_volume_1'], affine)
#save_nifti(output_directory + '/IvimDfixcyl_cyltest.nii.gz',ivim_fit_dmipy_fixed.fitted_parameters['C2CylinderStejskalTannerApproximation_1_lambda_par'], affine)

#save tensor
mu = fitted_params['C2CylinderStejskalTannerApproximation_1_mu']

"Returns the cartesian peak unit vectors of the model."
from dmipy.utils.utils import unitsphere2cart_Nd
mucart = unitsphere2cart_Nd(mu)

evecs_img = nib.Nifti1Image(mucart.astype(np.float32), img.affine)
nib.save(evecs_img, output_directory + '/tensor_evecs_cyltest.nii.gz')

# get the cylinder parameters and put them directly in the cylinder model to get
from dmipy.core.modeling_framework import MultiCompartmentModel


cyl_only_model = MultiCompartmentModel([cyl])
parameters_for_cyl_only = {'C2CylinderStejskalTannerApproximation_1_mu': fitted_params['C2CylinderStejskalTannerApproximation_1_mu'],
 'C2CylinderStejskalTannerApproximation_1_lambda_par': fitted_params['C2CylinderStejskalTannerApproximation_1_lambda_par'],
 'C2CylinderStejskalTannerApproximation_1_diameter': fitted_params['C2CylinderStejskalTannerApproximation_1_diameter']}

signal_cyl_only = cyl_only_model.simulate_signal(
    acquisition_scheme=scheme_ivim,
    parameters_array_or_dict=parameters_for_cyl_only)

print(signal_cyl_only.shape)
#save_nifti(output_directory+'/signal_cyl_only.nii.gz',signal_cyl_only,affine)

signal_cyl = nib.Nifti1Image(signal_cyl_only.astype(np.float32), img.affine)
nib.save(signal_cyl, output_directory + '/signal_cyl.nii.gz')

fcyl = input_directory + "/signal_cyl.nii.gz"
# Load the data with  mask
img = nib.load(fcyl)
img_data = img.get_fdata()
# define affine space
data1, affine = load_nifti(fcyl)

b = np.where(np.isnan(data1), 0, data1)

signal_cylb = nib.Nifti1Image(b.astype(np.float32), img.affine)
nib.save(signal_cylb, output_directory + '/signal_cylb.nii.gz')

# run the classical DTI model as reference
from dmipy.core.acquisition_scheme import gtab_dmipy2dipy
gtab = gtab_dmipy2dipy(scheme_ivim)
print(gtab)

dtimodel = dti.TensorModel(gtab)
dtifit = dtimodel.fit(b)
evecs = dtifit.evecs
# the principal eigenvector is our initial guess for the Stick model orientation
evecs_principal = evecs[..., 0]

#define classic FA and MD variables
dti_FA = dtifit.fa
dti_MD = dtifit.md
dti_RD = dtifit.rd
dti_AD = dtifit.ad


#save variables as nifti images

save_nifti(output_directory+'/cylFAtest.nii.gz',dti_FA,affine)
save_nifti(output_directory+'/cylMDtest.nii.gz',dti_MD,affine)
save_nifti(output_directory+'/cylRDtest.nii.gz',dti_RD,affine)
save_nifti(output_directory+'/cylADtest.nii.gz',dti_AD,affine)
save_nifti(output_directory+'/cyl_evecsptest.nii.gz',evecs_principal,affine)
save_nifti(output_directory+'/cyl_evecstest.nii.gz',evecs,affine)