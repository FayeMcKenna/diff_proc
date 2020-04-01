'''
============================================================
Bvec and Bval editing
============================================================
'''

#import modules
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy, acquisition_scheme_from_bvalues
import matplotlib.pyplot as plt
import numpy as np
from dipy.reconst.ivim import IvimModel
import nibabel as nib
import pathos
from dipy.segment.mask import applymask, bounding_box, crop
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import os

subjects = ['1001','1002','1003','1004','1006','1008','1009','1013','1014','1016','1017','1019','1020','1021','1022','1023','1024','1025','1026','1027','1028','1030','1031','1032','1033','1034','1035','1036','1037','1038','1039','1040','1041','1043','1044','1045','1048','1050','1052','1053','1054','1055','1056','1057','1058','1059','1060','1061','1062','1063','1064','1068','1067','1069','1070','1071','1072','1073','1074','1076','1077','1078','1079','1080','1081','1082','1083','1086','1087','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1201','1203','1206','1207','1208','1209','1210','1211','1212','1213','1214','1215','1216' ]
for subj in subjects:
    try:

        """
        Load bvec and bval files
        """

        # load directory names
        input_directory = "/Users/fayemckenna/Desktop/Ivim/ivim_proc/bfiles/%s" % subj
        output_directory = "/Users/fayemckenna/Desktop/Ivim/ivim_proc/bfiles/%s" % subj

        # filenames -- mask has all NaN removed to work with function
        fbval = input_directory + "/all_new_bvals.txt"
        fbvec = input_directory + "/ave_bvecs.txt"

        from dipy.io import read_bvals_bvecs
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)

        """
        IVIM
        """
        #select IVIM indices from bvecs
        ivim_bvecs = bvecs[[0,0,1,2,3,5,6,7,8,9,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,132,134,136,138,140,142,144,146,148,150],:]

        # transpose bvecs
        t_ivim_bvecs = np.transpose(ivim_bvecs)

        #select IVIM indices from bvals
        nbvals = np.array(bvals)
        nbvals[0:10]=0
        indices = [0,0,1,2,3,5,6,7,8,9,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,132,134,136,138,140,142,144,146,148,150]
        ivim_bvals = np.take(nbvals, indices)
        print(ivim_bvals)

        #traspose bvals
        a = np.array(ivim_bvals)
        a = a.reshape((-1,1))
        t_ivim_bvals = np.transpose(a)

        #check dimensions match
        print(t_ivim_bvecs.shape)
        print(t_ivim_bvals.shape)

        """
        NODDI
        """

        #select IVIM indices from bvecs
        noddi_bvecs = bvecs[[0,0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150],:]

        # transpose bvecs
        t_noddi_bvecs = np.transpose(noddi_bvecs)

        #select IVIM indices from bvals
        n2bvals = np.array(bvals)
        n2bvals[0:10]=0
        indices2 = [0,0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150]
        noddi_bvals = np.take(n2bvals, indices2)
        print(noddi_bvals)

        #traspose bvals
        b = np.array(noddi_bvals)
        b = b.reshape((-1,1))
        t_noddi_bvals = np.transpose(b)

        #check dimensions match
        print(t_noddi_bvecs.shape)
        print(t_noddi_bvals.shape)

        """
        Export
        """

        #save files
        np.savetxt(output_directory + "/ivimbvecs.txt", np.array(t_ivim_bvecs), fmt="%s")
        np.savetxt(output_directory + "/ivimbvals.txt", np.array(t_ivim_bvals), fmt="%s")
        np.savetxt(output_directory + "/noddibvecs.txt", np.array(t_noddi_bvecs), fmt="%s")
        np.savetxt(output_directory + "/noddibvals.txt", np.array(t_noddi_bvals), fmt="%s")

    except Exception as error:
        print(error)