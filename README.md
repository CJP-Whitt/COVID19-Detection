Project for finding cases of possible COVID19 induced viral pneumonia.


# Training/Validation Dataset
Images set used for all training and validation available here: https://drive.google.com/file/d/1xn-7_PvUr9fiCDt0f6hCBcftW2ZG2TXo/view?usp=sharing

To run most of the files, you would have to stay inside the COVID19_Detection directory on MATLAB and access 
the other files by clicking the + symbol next to the folders and then clicking the file from there.

SURF_SVM.m: extracts features using the SURF method.
splitData.mlx: Splits the UCSD data into training, testing, and validation directories.
pham_code.mlx: The code the nature overview paper used for transfer learning with a 80/20 random data split.
leaveOneOutSetup.m: Put's one patient's images into a testing directory and the rest into a training directory.
uscd_Data: contains testing/training/validation data splits for UCSD data.
UCSD_combined: contains all of UCSD data in one training folder.
transferLearning_ucsdSplit: Preliminary transfer learning files for 4 CNNs using a data split that was provided on UCSD dataset github.
transferLearning_ucsdRandomSplit: Preliminary transfer learning files using random 80/20 data split.
transferLearning_ucsdLeaveOneOut: Preliminary transfer learning files using leave one patient out cross validation.
images_and_split: contains all COVID and nonCOVID images from UCSD data.
finalTesting: Contains transfer learning and SVM testing files that was used for final dataset.
featureExtraction_ucsdLeaveOneOut: SVM feature extraction files using leave one patient out cross validation.
COVID19_CT_seg_20cases: Contains final testing set with 3 images from each patient from CT scans.