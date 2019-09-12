"""
Author: Shazia Akbar

File: bda.py
Description: Novel method for simultaneously sparsely exploring large
    WSI and then densely sampling regions of interest
    [Currently in development]
"""

import tensorflow as tf

sess = tf.Session()

import numpy as np
import glob
import os
import utils
import time
import math

from keras.models import Model
from keras.utils import multi_gpu_model, to_categorical
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.models import load_model
from keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from keras.optimizers import Adam, SGD, Nadam
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from se_inception_v3 import SEInceptionV3

import scipy
from scipy.misc import imread
from skimage.morphology import binary_dilation, disk
import matplotlib.pyplot as plt
from openslide.lowlevel import OpenSlideError
from skimage.morphology import binary_dilation, disk
from scipy.ndimage import sobel, uniform_filter
from skimage.color import rgb2hed
from scipy.ndimage.interpolation import rotate

from keras import backend as K
K.set_image_data_format('channels_first')

config = tf.ConfigProto(allow_soft_placement=True)
K.set_session(tf.Session(config=config))

import matplotlib.pyplot as plt


# Define global parameters
RESOLUTION = 1									# level in pyramid from which to extract patches
NUM_GPU = 1										# number of GPUs to use in experiment
BLOCK = 10										# parameter for determining how many slides can be loading into memory at once			
COARSE_STEP = 2048							# step size for the sparse selection
LOSS_TYPE = 'categorical_crossentropy'	# loss function for CNN			


# Better method for saving multiGPU checkpoints
class ModelMGPU(Model):
	def __init__(self, ser_model, gpus):
		pmodel = multi_gpu_model(ser_model, gpus)
		self.__dict__.update(pmodel.__dict__)
		self._smodel = ser_model

	def __getattribute__(self, attrname):
		'''Override load and save methods to be used from the serial-model. The
		serial-model holds references to the weights in the multi-gpu model.
		'''
		# return Model.__getattribute__(self, attrname)
		if 'load' in attrname or 'save' in attrname:
				return getattr(self._smodel, attrname)

		return super(ModelMGPU, self).__getattribute__(attrname)



def get_duct_mask(_filename):
	"""
	Mask implementation has been removed for privacy purposes
	"""
	
	return None
	
			
def selection_criteria(patient_file, locations, patch_size, selection_type="random"):
	"""
	Method for extracting patches from whole slide images - we may change this to use something more elaborate than
	random selection later
	
	:param patient_file: path to svs file
	:param locations: list of coordinates to pass to extract algorithm
	:param patch_size: (width, height) of each patch
	:param selection_type: "random" is the only option for now
	:return: patches in numpy array
	"""
	if selection_type is "random":
		return utils.get_patch_from_locations(patient_file, locations, patch_size,
															resolution_level=RESOLUTION), locations


def initialize_static_attention(dims, mask=None, variance_shift=(256, 256), randomize=True):
	"""
	Method for create a grid across the dimensions of the whole slide image
	
	: param dims: original (width, height) dimensions of WSI
	: param mask: optional mask to ignore parts of image
	: param variance_shift: determine by how much points are randomly shift up/down/left/right
	: param randomize: set to True if using variance_shift
	"""
	
	# Initialize regular (sparse) grid
	X, Y = np.mgrid[COARSE_STEP * 2: dims[0] - (COARSE_STEP * 2):COARSE_STEP, 
			COARSE_STEP * 2:dims[1] - (COARSE_STEP * 2):COARSE_STEP]
			
	coordinates_twod = np.concatenate((X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]), axis=1).astype('int32')

	# Random shifts in x and y direction
	if randomize == True:
		coordinates_twod = coordinates_twod.transpose(1, 0)
		coordinates_twod[0] = coordinates_twod[0] + ((np.random.random_sample() * variance_shift[0] * 2) - variance_shift[0])
		coordinates_twod[1] = coordinates_twod[1] + ((np.random.random_sample() * variance_shift[1] * 2) - variance_shift[1])
		coordinates_twod = coordinates_twod.transpose(1, 0)
		
	if mask is not None:
		# mask out regions outside duct
		masked_coordinates = []
		mask_scale_x = float(mask.shape[1]) / dims[0]
		mask_scale_y = float(mask.shape[0]) / dims[1]
		
		for point in coordinates_twod:
			if mask[int(point[1] * mask_scale_x), int(point[0] * mask_scale_y)] == 1:
				masked_coordinates.append(point)
		
		'''
		# add some random locations in addition to ducts (same number)		
		rand_idx = np.arange(len(coordinates_twod))
		np.random.shuffle(rand_idx)
		for i in range(len(masked_coordinates)):
			masked_coordinates.append(coordinates_twod[rand_idx[i]])
		'''
		
		coordinates_twod = np.array(masked_coordinates)
		
	return coordinates_twod
	
def get_total_sample_patches_in_train(training_files):
	"""
	Method for computing number of training instances prior to training
	"""
	i = 0
	for _filename in training_files:
		image_dims = utils.get_image_dims(_filename, resolution_level=RESOLUTION)
		mask = get_duct_mask(_filename)
		
		lookup_loc = initialize_static_attention(image_dims, mask=mask, randomize = False)
		
		i += len(lookup_loc)
	return i
		
def get_random_coordinates(list_filenames, topk):
	"""
	Method added later to experiment with locations selected at random from around WSI
	"""
	locations = []

	for idx, _filename in enumerate(list_filenames):
		image_dims = utils.get_image_dims(_filename, resolution_level=RESOLUTION)
		_mask = get_duct_mask(_filename)
		
		lookup_loc = initialize_static_attention(image_dims, mask = _mask, randomize = True)
		ind = np.arange(len(lookup_loc))
		np.random.shuffle(ind)
		locations.append(lookup_loc[ind[:topk]])
		
	return locations	
	
def get_attention_map_for_files(model, list_filenames, recurrence_labels, patch_size, topk):
	"""
	Method for creating "attention maps" for WSI
	Here, an attention map is the predictions generated by the network thus far (midway through training)
	
	:param model: the trained model from which predictions are retrieved
	:param list_filenames: list of training data 
	:param recurrence_labels: ground truth labels same size as list_filenames
	:param patch_size
	:param topk: number of locations of interest extracted for dense sampling
	"""
	
	locations = []
	model = Model([model.input], model.layers[-2].output)

	for idx, _filename in enumerate(list_filenames):
		#print(_filename)
		
		# get predictions - stepwise
		image_dims = utils.get_image_dims(_filename, resolution_level=RESOLUTION)
		_mask = get_duct_mask(_filename)
		
		lookup_loc = initialize_static_attention(image_dims, mask = _mask, randomize = False)	# retrieve a regular grid and densely sample around
		#print(lookup_loc.shape)
		p = []
		
		for l in range(0, len(lookup_loc), 100):
			patches_from_file, _ = selection_criteria(_filename, lookup_loc[l:min(l + 100, len(lookup_loc))], patch_size)
			
			if len(patches_from_file) > 0:
				_p = model.predict(patches_from_file).mean(axis=1)
				
				'''
				if recurrence_labels[idx] == 1:
					_p = model.predict(patches_from_file)[:, 1]
				else:
					_p = model.predict(patches_from_file)[:, 0]
				'''
								
				if len(p) == 0:
					p = _p
				else:
					p = np.concatenate((p, _p), axis=0)
		
		'''
		# sort predictions according to label
		if recurrence_labels[idx] == 1:
			sort_idx = np.argsort(np.array(p))[::-1]
		else:
			sort_idx = np.argsort(np.array(p))
		'''
		sort_idx = np.argsort(np.array(p))[::-1]
		
		locations.append(lookup_loc[sort_idx[:topk]])
	
	#print(locations[0], len(locations))
	
	return locations


def minibatch_gen_attention(train_files, train_seg_labels, batch_size, patch_size):
	"""
	Keras generator for learning attention maps per image

	:param train_files: list of paths to training data
	:param train_seg_labels: list of paths to segmentation for ground truth
	:param batch_size:
	:param patch_size: (width, height) of each patch
	:param randomized_locs: list of valid (x, y) coordinates to lookup in whole slides
	:return: (patches, labels) per batch
	"""
	while True:
		for j in range(0, len(train_files), BLOCK):
			x, y = [], []
			for idx, _filename in enumerate(train_files[j:min(len(train_files), j + BLOCK)]):
	
				# build sample map 
				image_dims = utils.get_image_dims(_filename, resolution_level=RESOLUTION)
				mask = get_duct_mask(_filename)
				sampling = initialize_static_attention(image_dims, mask=mask, randomize=True)
				
				# We are using the same model for attention and task therefore need to keep sampling the same i.e. at same res
				patches_from_file, _ = selection_criteria(_filename, sampling, patch_size)
	
				# duplicating label
				if len(patches_from_file) > 0:
					labels = np.tile(int(train_seg_labels[j+idx]), len(patches_from_file)).astype('int32')
					labels = to_categorical(labels, 2)
	
					if len(x) == 0:
						x, y = patches_from_file, labels
					else:
						x = np.concatenate((x, patches_from_file))
						y = np.concatenate((y, labels))
						
			#print(x.shape, y.shape)
	
			indices = np.arange(len(y))
			np.random.shuffle(indices)
			for start_idx in range(0, len(y) - batch_size + 1, batch_size):
				yield x[indices[start_idx: start_idx + batch_size]], \
						y[indices[start_idx: start_idx + batch_size]]
					
							
def minibatch_gen_task(train_files, train_seg_labels, batch_size, patch_size, attention_locations):
	"""
	Keras generator for supervised learning approach

	:param train_files: list of paths to training data
	:param train_seg_labels: list of paths to segmentation for ground truth
	:param batch_size:
	:param patch_size: (width, height) of each patch
	:param randomized_locs: list of valid (x, y) coordinates to lookup in whole slides
	:return: (patches, labels) per batch
	"""

	_x = np.arange(-(COARSE_STEP / 2), (COARSE_STEP / 2) - (patch_size[0]), patch_size[0])
	_y = np.arange(-(COARSE_STEP / 2), (COARSE_STEP / 2) - (patch_size[1]), patch_size[1])
	xv, yv = np.meshgrid(_x, _y, copy=False)
	grid = np.concatenate((xv.ravel()[:, np.newaxis], yv.ravel()[:, np.newaxis]), axis=1).astype('int32')

	while True:
		
		for j in range(0, len(train_files), BLOCK):
			x, y = np.array([]), np.array([])
			for idx, _filename in enumerate(train_files[j:min(len(train_files), j+BLOCK)]):
				
				image_dims = utils.get_image_dims(_filename, resolution_level=RESOLUTION)

				#modify sparse attention coordinates
				new_attentions = np.zeros((0, 2), dtype='int32')
				
				for a in range(len(attention_locations[j+idx])):
					#print(attention_locations[j+idx][a], image_dims)
					_xc, _xy = attention_locations[j+idx][a]
					new_attentions = np.concatenate((new_attentions, grid + attention_locations[j+idx][a : a+1]), axis=0)

				#print(attention_locations[j+idx], _filename)
				# locations may be different depending on the selection criteria
				try:
					patches_from_file, new_locations = selection_criteria(_filename, new_attentions, patch_size)
				except OpenSlideError:
					print(attention_locations[j+idx], _filename)

				# duplicating label
				if len(patches_from_file) > 0:

					labels = np.tile(int(train_seg_labels[j+idx]), len(patches_from_file)).astype('int32')
					labels = to_categorical(labels, 2)

					if len(x) == 0:
						x, y = patches_from_file, labels
					else:
						#print patches_from_file
						x = np.concatenate((x, patches_from_file))
						y = np.concatenate((y, labels))

			indices = np.arange(len(y))
			np.random.shuffle(indices)
			for start_idx in range(0, len(y) - batch_size + 1, batch_size):
				yield x[indices[start_idx: start_idx + batch_size]], \
						y[indices[start_idx: start_idx + batch_size]]


# to store the loss per batch - not just per epoch
class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


class CNNAlgorithm():
	"""
	Main class for performing an experiment on whole slide images
	"""
	def __init__(self, batch_size, num_epochs, shape_size, mask_locations, cnntype='inceptionSE', loss_type='decay', topk = 5):
		"""
		__init__

		:param autoencoder_loc: path to VAE model
		:param batch_size:
		:param num_epochs:
		:param shape_size: (width, height, channel) of input data
		:param mask_locations: path to locations of segmentation ground truth (needed for supervised method)
		:param cnntype: type of cnn model to train
		
		"""
		self.batch_size = batch_size
		self.shape = shape_size
		self.num_epochs = num_epochs
		self.segmentation_masks = mask_locations
		self.cnn_type = cnntype
		self.loss_type = loss_type
		self.topk = topk
		
	def get_model(self):
		"""
		Build deep neural network to perform classification task
		:return: Keras model
		"""
		input_img = Input(shape=self.shape, name='input_data')
		
		if self.cnn_type == "inception":
			i3 = InceptionV3(weights="imagenet", include_top=False, input_tensor=input_img)
			cnn_model = i3.output
			cnn_model = GlobalAveragePooling2D()(cnn_model)
			cnn_model = BatchNormalization()(cnn_model)
			cnn_model = Dropout(0.5)(cnn_model)
		elif self.cnn_type == "dense":
			i3 = DenseNet121(weights="imagenet", include_top=False, input_tensor=input_img)
			cnn_model = i3.output
			cnn_model = GlobalAveragePooling2D()(cnn_model)
		elif self.cnn_type == "resnet":
			i3 = ResNet50(weights="imagenet", include_top=False, input_tensor=input_img)
			cnn_model = i3.output
			cnn_model = GlobalAveragePooling2D()(cnn_model)
		elif self.cnn_type == "inceptionSE":
			i3 = SEInceptionV3(weights=None, include_top=False, input_tensor=input_img)
			cnn_model = i3.output
			cnn_model = GlobalAveragePooling2D()(cnn_model)
			#cnn_model = BatchNormalization()(cnn_model)
			#cnn_model = Dropout(0.2)(cnn_model)
			
		else:
			raise Exception('Model type specified not implemented')

		dense1 = Dense(512, name='dense1')(cnn_model)
		output = Dense(2, activation='softmax', name='p_out')(dense1)
		
		model = Model([input_img], output)
		print(model.summary())
		
		return model
		
   
	def train(self, training_files, training_labels, validation_files, validation_labels, cnn_model_location, start_idx=0, decay_rate = 0.001):
		"""
		Main training function

		:param training_files: list of paths to training data
		:param training_labels: list of image-level labels for training data
		:param cnn_model_location: path and name to save the model once trained
		:param start_idx: current epoch (if using job arrays)
		:return:
		"""

		assert len(training_files) > 0
		assert len(training_files) == len(training_labels)

		patch_size = (self.shape[-2], self.shape[-1])
		
		adjusted_epochs = self.num_epochs - start_idx
		total1_patches = get_total_sample_patches_in_train(training_files)
		total1_val_patches = get_total_sample_patches_in_train(validation_files)
		
		_x = np.arange(-(COARSE_STEP / 2) , (COARSE_STEP / 2) - (patch_size[0]), patch_size[0])
		_y = np.arange(-(COARSE_STEP / 2) , (COARSE_STEP / 2) - (patch_size[1]), patch_size[1])
		num_dense_samples = len(_x) * len(_y)
		total2_patches = num_dense_samples * self.topk * len(training_files) 
		total2_val_patches = num_dense_samples * self.topk * len(validation_files) 
		print(total1_patches, total2_patches)
		
		# Load weight from previous epoch
		#with tf.device('/cpu:0'):
		base_model = self.get_model()
		if os.path.isfile(cnn_model_location + "_checkpoints/" + 'weights.' + str(start_idx-1) + '.h5'):
			print("Loading previous weights")
			base_model.load_weights(cnn_model_location + "_checkpoints/" + 'weights.' + str(start_idx-1) + '.h5')

		# Wrap parallel model if needed
		if NUM_GPU > 1:
				parallel_model = ModelMGPU(base_model, gpus=NUM_GPU)
		else:
				parallel_model = base_model

		parallel_model.compile(loss=LOSS_TYPE, optimizer='sgd')
		history = LossHistory()
		
		checkpoint_dir = cnn_model_location + "_checkpoints"
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
			
		# fix for multi-gpu
		class MyCbk(Callback):
			def __init__(self, model):
				self.model_to_save = model
				self.count = start_idx

			def on_epoch_end(self, epoch, logs=None):
				self.model_to_save.save(checkpoint_dir + "/weights." + str(self.count) + ".h5")
				self.count = self.count + 1
				
		cbk = MyCbk(base_model)		

		start_time = time.time()
		#print(base_model.summary())
		
		def step_decay(current_epoch, current_lr):
			drop = 0.2
			epochs_drop = 2.0
			lrate = current_lr * math.pow(drop, math.floor((1 + current_epoch) / epochs_drop))
			return lrate
			
		# Initialize two different learning rates for two stages
		# TODO: grid seach learning rate
		if self.loss_type == 'two_losses':
			sparse_lr = 0.000001
			dense_lr = 0.001
		else:
			sparse_lr = 0.01
			dense_lr = 0.01
		
		# Specify decay rate with respect to number of epochs
		decay_rate = (1. / (1. + decay_rate * self.num_epochs))
		
		for i in range(0, start_idx):
			if i != 0:
				sparse_lr = sparse_lr * decay_rate
				dense_lr = dense_lr * decay_rate
		
		
		for i in range(start_idx, self.num_epochs):
			# Set new learning rate
			if i != 0:
				if self.loss_type == 'decay':
					sparse_lr = step_decay(sparse_lr)
				else:
					sparse_lr = sparse_lr * decay_rate
			K.set_value(parallel_model.optimizer.lr, sparse_lr)
			
			print("New learning rate: " + str(K.get_value(parallel_model.optimizer.lr)))
				
			# Stage 1: Coarse sampling of whole slide image
			print("Coarse sampling... epoch " + str(i) + "/" + str(self.num_epochs))
			
			# Update attention map
			parallel_model.fit_generator(
					minibatch_gen_attention(training_files, training_labels, self.batch_size, patch_size),
					steps_per_epoch=total1_patches // self.batch_size,
					#validation_data = minibatch_gen_attention(validation_files, validation_labels, self.batch_size, patch_size),
					#validation_steps = total1_val_patches // self.batch_size,
					epochs=1, verbose=1, callbacks=[history]
					#, class_weight={0:0.5, 1:1.0}
				)
		
			print("Retrieving maps...")
			
			# Run current model and retrieve list of locations to lookup ordered by attention response
			attention_coords = get_attention_map_for_files(parallel_model, training_files, training_labels, patch_size, self.topk)
			attention_coords_val = get_attention_map_for_files(parallel_model, validation_files, validation_labels, patch_size, self.topk)
			
			if i != 0:
				if self.loss_type == 'decay':
					dense_lr = step_decay(dense_lr)
				else:
					dense_lr = dense_lr * decay_rate
			K.set_value(parallel_model.optimizer.lr, dense_lr)
			
			# Stage 2: Dense sampling in areas of interest defined my attention map 
			print("Dense sampling...")
			parallel_model.fit_generator(
					minibatch_gen_task(training_files, training_labels, self.batch_size, patch_size, attention_coords),
					steps_per_epoch=total2_patches // self.batch_size,
					validation_data = minibatch_gen_task(validation_files, validation_labels, self.batch_size, patch_size, attention_coords_val),
					validation_steps = total2_val_patches // self.batch_size,
					epochs=1, verbose=1, callbacks=[history, cbk]
					#, class_weight={0:0.5, 1:1.0}
				)
				
		print("completed in {:.3f}s".format(time.time() - start_time))
    
		# Save temporary model after every epoch
		base_model.save(cnn_model_location + '.weights.' + str(start_idx) + '.h5')
		np.savetxt(cnn_model_location + '.history.' + str(start_idx) + '.txt', np.array(history.losses), delimiter=',')
