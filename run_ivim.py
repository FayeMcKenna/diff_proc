'''
============================================================
Intravoxel incoherent motion dmipy
============================================================
'''

#import modules
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy, acquisition_scheme_from_bvalues
import matplotlib.pyplot as plt
import numpy as np
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
#['1001','1002','1003','1004','1006','1008','1009','1013','1014','1016','1017','1019','1020','1021','1022','1023','1024','1025','1026','1027','1028','1030','1031','1032','1033','1034','1035','1036','1037','1038','1039','1040','1041','1043','1044','1045','1048','1050','1052','1053','1054','1055','1056','1057','1058','1059','1060','1061','1062','1063','1064','1068','1067','1069','1070','1071','1072','1073','1074','1076','1077','1078','1079','1080','1081','1082','1083','1086','1087','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1201','1203','1206','1207','1208','1209','1210','1211','1212','1213','1214','1215','1216' ]

subjects = ['1024','1025','1030','1031','1032','1037','1038','1040','1041','1043','1044','1045','1039','1048','1050','1052','1055','1056','1061','1062','1063','1064','1067','1068','1069','1070','1071','1072','1073','1074','1076','1077']
for subj in subjects:
    try:
        """
        Load data and set mask 
        """
        # load directory names
        input_directory = "/gpfs/home/fm1545/Faye/Diff_models/toprocess/%s" %subj
        output_directory = "/gpfs/home/fm1545/Faye/Diff_models/toprocess/%s" %subj

        # filenames -- mask has all NaN removed to work with function
        fivim = input_directory + "/ivimdata.nii.gz"
        fmask = input_directory + "/mask.nii.gz"
        fbval = input_directory + "/ivimbvals.txt"
        fbvec = input_directory + "/ivimbvecs.txt"

        # load bvals, bvecs and gradient files
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)

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

        scheme_ivim = gtab_dipy2dmipy(gtab, b0_threshold=1e6, min_b_shell_distance=1e7)
        scheme_ivim.print_acquisition_info

        # fit model
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
        Parameter map for dmipy model
        """
        from dmipy.custom_optimizers.intra_voxel_incoherent_motion import ivim_Dstar_fixed
        from time import time

        ivim_fit_dmipy_fixed = ivim_Dstar_fixed(scheme_ivim, data)

        save_nifti(output_directory + '/IvimDfixperf.nii.gz',
                   ivim_fit_dmipy_fixed.fitted_parameters['partial_volume_1'], affine)
        save_nifti(output_directory + '/IvimDfixdiff.nii.gz',
                   ivim_fit_dmipy_fixed.fitted_parameters['G1Ball_1_lambda_iso'], affine)
    except Exception as error:
        print(error)