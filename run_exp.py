import pickle
import glob
import utils
import os
import sys
from random import shuffle
import numpy as np

from bda import CNNAlgorithm as alg


###########################################################################################
# Load training data

shape_size = (3, 256, 256)

slide_locations = '/labs3/amartel_data3/histology/Data/DCIS_cohort/DCIS_Training_Dataset/'
train, val, test = utils.load_dcis_labels('/labs3/amartel_data3/histology/Data/DCIS_cohort/DCIS_Training_Dataset/CaseSetsDefined.csv')

print("training_samples: ", len(train[0]))

train_files = [os.path.join(slide_locations, str(int(x))) + '.svs' for x in train[0]]
val_files = [os.path.join(slide_locations, str(int(x))) + '.svs' for x in val[0]]

# shuffle train_files
combined = list(zip(train_files, train[1]))
np.random.shuffle(combined)
train_files, train_labels = zip(*combined)

###########################################################################################
# Run BDA experiment

num_epochs = 100
batch_size = 32
start_epoch = 0
modelname = 'attention-task_top5_sparse2048_switchtwolossv2_pyr1_dense_batch32_3class'
loss_type = 'two_loss'		# two_loss/decay/single
cnn = 'dense'		# inceptionSE/inception

cnn_model_location = './models/' + modelname
segmentation_locations = '/labs3/amartel_data3/histology/Data/DCIS_cohort/DCIS_Training_Dataset/dcis_seg/'

alg_instance = alg(batch_size = batch_size,
                   num_epochs = num_epochs,
                   shape_size = shape_size,
                   mask_locations = segmentation_locations,
                   cnntype = cnn,
                   loss_type = loss_type)


alg_instance.train(train_files, train_labels, val_files, val[1], cnn_model_location, start_idx=start_epoch)

