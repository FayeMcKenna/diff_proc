'''
============================================================
Neurite Orientation and Distribution dmipy
============================================================
'''

#import modules
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy, acquisition_scheme_from_bvalues
import matplotlib.pyplot as plt
import numba as np
import numpy
from os.path import join
from dipy.reconst.ivim import IvimModel
import nibabel as nib
from dipy.segment.mask import applymask, bounding_box, crop
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


#['1001','1002','1003','1004','1006','1008','1009','1013','1014','1016','1017','1019','1020','1021','1022','1023','1024','1025','1026','1027','1028','1030','1031','1032','1033','1034','1035','1036','1037','1038','1039','1040','1041','1043','1044','1045','1048','1050','1052','1053','1054','1055','1056','1057','1058','1059','1060','1061','1062','1063','1064','1068','1067','1069','1070','1071','1072','1073','1074','1076','1077','1078','1079','1080','1081','1082','1083','1086','1087','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1201','1203','1206','1207','1208','1209','1210','1211','1212','1213','1214','1215','1216' ]

subjects = ['1001']

for subj in subjects:
    try:
        """
        Load data and set mask 
        """
        # load directory names
        input_directory = "/Users/fayemckenna/Desktop/ivim/Diff_procnew/%s" %subj
        output_directory = "/Users/fayemckenna/Desktop/ivim/Diff_procnew/%s" %subj

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
        NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

        NODDI_fit_hcp = NODDI_mod.fit(acq_scheme, data, mask=mask)

        fitted_parameters = NODDI_fit_hcp.fitted_parameters

        # get total Stick signal contribution
        vf_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] *
                    fitted_parameters['partial_volume_1'])
        # get total Zeppelin signal contribution
        vf_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) *
                    fitted_parameters['partial_volume_1'])

        """
        Save output
        """
        save_nifti(output_directory + '/NoddiODI.nii.gz',
                   NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi'], affine)
        save_nifti(output_directory + '/NoddiISO.nii.gz', NODDI_fit_hcp.fitted_parameters['G1Ball_1_lambda_iso'],
                   affine)
        save_nifti(output_directory + '/Noddistick.nii.gz',
                   NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'], affine)
        save_nifti(output_directory + '/Noddiwm.nii.gz', NODDI_fit_hcp.fitted_parameters['partial_volume_1'], affine)
        save_nifti(output_directory + '/Noddicsf.nii.gz', NODDI_fit_hcp.fitted_parameters['partial_volume_0'], affine)
        save_nifti(output_directory + '/Noddivfintra.nii.gz', NODDI_fit_hcp.fitted_parameters['Vf_intra'], affine)
        save_nifti(output_directory + '/Noddivfextra.nii.gz', NODDI_fit_hcp.fitted_parameters['Vf_extra'], affine)
        save_nifti(output_directory + '/Noddivfintra2.nii.gz', Vf_intra, affine)
        save_nifti(output_directory + '/Noddivfextra2.nii.gz', Vf_extra, affine)

    except Exception as error:
        print(error)