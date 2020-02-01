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

subjects = ['1004','1006','1008','1009','1013','1016','1017','1019','1020','1021','1022','1023','1024','1025','1026','1027','1028','1029','1030','1032','1033','1035','1037','1038','1040','1041','1043_1','1044','1050','1052','1055','1061','1062','1063','1064','1067','1068','1069','1070','1072','1073','1074','1076','1077','1078','1079','1080','1081','1082','1083','1086','1087','1088','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1201','1203','1206','1208','1209','1210','1211','1212','1213','1214','1215','1216']
for subj in subjects:
    # load directory names
    input_directory = "/Users/fm1545/Desktop/R01_study/study/%s/dtrecon/output" % subj
    #working_directory = "data/working"
    output_directory = "/Users/fm1545/Desktop/R01_study/study/%s/dtrecon/output" % subj

    # filenames -- mask has all NaN removed to work with function
    fdwi = input_directory + "/FWIdata.nii.gz"
    fmask = input_directory + "/mask.nii.gz"
    fbval = input_directory + "/fwibvals.txt"
    fbvec = input_directory + "/fwibvecs.txt"

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
    save_nifti(output_directory+'/WB_regMD.nii.gz',dti_MD,affine)
    save_nifti(output_directory+'/WB_regFA.nii.gz',dti_FA,affine)
    save_nifti(output_directory+'/WB_fwi.nii.gz',F,affine)
    save_nifti(output_directory+'/WB_fwiFA.nii.gz',FA,affine)
    save_nifti(output_directory+'/WB_fwiMD.nii.gz',MD,affine)


