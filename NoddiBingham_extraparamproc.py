
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
from dmipy.distributions.distributions import odi2kappa
import math
from math import tan
from math import pi

#['1001','1002','1003','1004','1006','1008','1009','1013','1014','1016','1017','1019','1020','1021','1022','1023','1024','1025','1026','1027','1028','1030','1031','1032','1033','1034','1035','1036','1037','1038','1039','1040','1041','1043','1044','1045','1048','1050','1052','1053','1054','1055','1056','1057','1058','1059','1060','1061','1062','1063','1064','1068','1067','1069','1070','1071','1072','1073','1074','1076','1077','1078','1079','1080','1081','1082','1083','1086','1087','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1201','1203','1206','1207','1208','1209','1210','1211','1212','1213','1214','1215','1216' ]

subjects = ['1001','1002','1003','1004','1006','1008','1009','1013','1014','1016','1017','1019','1020','1021','1022','1023','1024','1025','1026','1027','1028','1030','1031','1032','1033','1034','1035','1036','1037','1038','1039','1040','1041','1043','1044','1045','1048','1050','1052','1053','1054','1055','1056','1057','1058','1059','1060','1061','1062','1063','1064','1068','1067','1069','1070','1071','1072','1073','1074','1076','1077','1078','1079','1080','1081','1082','1083','1086','1087','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1201','1203','1206','1207','1208','1209','1210','1211','1212','1213','1214','1215','1216' ]

for subj in subjects:
    try:

        """
        Load data and set mask 
        """
        # load directory names
        input_directory = "/Users/fayemckenna/Desktop/Ivim/Diff_procnew/%s" %subj
        output_directory = "/Users/fayemckenna/Desktop/Ivim/Diff_procnew/%s" %subj

        # filenames -- mask has all NaN removed to work with function
        fodi = input_directory + "/NoddiB_ODI.nii.gz"
        fbeta = input_directory + "/NoddiB_betafrac.nii.gz"
        fmask = input_directory + "/mask.nii.gz"
        fbval = input_directory + "/noddibvals.txt"
        fbvec = input_directory + "/noddibvecs.txt"

        # Load the data with  mask
        img = nib.load(fodi)
        img_data = img.get_fdata()
        img.header.get_data_shape()
        empty_header = nib.Nifti1Header()
        empty_header.get_data_shape()
        # define affine space
        odi, affine = load_nifti(fodi)


        # Load the data with  mask
        img = nib.load(fbeta)
        img_data = img.get_fdata()
        # define affine space
        betafrac, affine = load_nifti(fbeta)

        """
        Run Model
        """
        # kappa = 1./(tan(0.5*pi*odi));
        kappa = odi2kappa(odi)
        beta = betafrac*kappa

        ODIp = np.arctan2(1.0, kappa - beta) * 2 / np.pi
        ODIs= np.arctan2(1.0, kappa) * 2 / np.pi
        ODIt = np.arctan2(1.0, np.sqrt(np.abs(kappa * (kappa - beta)))) * 2 / np.pi
        DAI= np.arctan2(beta, kappa - beta) * 2 / np.pi

        #ODIp
        ODIp = np.array(ODIp)
        odip_img = nib.Nifti1Image(ODIp, img.affine, empty_header)
        odip_img.header.get_data_shape()

        #ODIS
        ODIs = np.array(ODIs)
        odis_img = nib.Nifti1Image(ODIs, img.affine, empty_header)
        odis_img.header.get_data_shape()

        #ODIt
        ODIt = np.array(ODIt)
        odit_img = nib.Nifti1Image(ODIt, img.affine, empty_header)
        odit_img.header.get_data_shape()

        #DAI
        DAI = np.array(DAI)
        dai_img = nib.Nifti1Image(DAI, img.affine, empty_header)
        dai_img.header.get_data_shape()


        nib.save(odip_img, output_directory + 'Noddib_ODIp.nii.gz')
        nib.save(odis_img, output_directory + 'Noddib_ODIs.nii.gz')
        nib.save(odit_img, output_directory + 'Noddib_ODIt.nii.gz')
        nib.save(odis_img, output_directory + 'Noddib_DAI.nii.gz')

    except Exception as error:
        print(error)