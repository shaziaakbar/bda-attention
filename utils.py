import sys

sys.path.insert(0, './extract')

import extract
import numpy as np
import pandas
from openslide import open_slide
import matplotlib.pyplot as plt
try:
	from skimage import filters
except ImportError:
	from skimage import filter as filters
from scipy.misc import imresize
import matplotlib

from skimage.draw import polygon_perimeter
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import regionprops
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from skimage.morphology import disk


def get_image_dims(filename, resolution_level=0):
	_slide = open_slide(filename)
	dims = _slide.level_dimensions[resolution_level]
	_slide.close()
	return dims
	

# retrieve only patches containing tissue from filename alongwith the location of those patches
def get_tissue_patch_locations_from_tif(filename, tile_size, resolution_level=0, version=2):
	_slide = open_slide(filename)
	orig_dims = _slide.dimensions
	_slide.close()
	tissue_mask = _get_tissue_mask(filename, version=version)
	
	'''
	plt.subplot(121)
	plt.imshow(downscale_image[:,:,0])
	plt.subplot(122)
	plt.imshow(tissue_mask)
	plt.colorbar()
	plt.show()
	'''

	einst = extract.TissueLocator(filename, tile_size, level=resolution_level, mode="all", mask=tissue_mask)
	location_of_patches = einst.get_coordinates_as_list(orig_dims)
	return location_of_patches
	

#NOTE: when you read a region from ANY level in the pyramid, the location MUST be at level 0
def get_tissue_patch_locations_from_tif_custom(filename, tile_size, tissue_mask, resolution_level=0):
	_slide = open_slide(filename)
	orig_dims = _slide.level_dimensions[resolution_level]
	_slide.close()
	
	einst = extract.TissueLocator(filename, tile_size, level=resolution_level, mode="all", mask=tissue_mask)
	location_of_patches = einst.get_coordinates_as_list(orig_dims)
	return location_of_patches

def get_patch_from_locations(filename, locations, patch_size, resolution_level=0):
	#	convert locations to full res
	full_res_locations= []
	for i in range(len(locations)):
		if resolution_level == 0:
			full_res_locations.append([locations[i, 0] * (2 ** (resolution_level)), locations[i, 1] * (2 ** (resolution_level))])
		else:
			full_res_locations.append([locations[i, 0] * (2 ** (resolution_level + 1)), locations[i, 1] * (2 ** (resolution_level + 1))])
	full_res_locations = np.asarray(full_res_locations)

	einst = extract.TissueLocator(filename, patch_size, level = resolution_level, mode="all")
	image_as_patches = einst.get_tissue_patches(full_res_locations)
	return image_as_patches


# features to read CSV file and create list of training, validation, test files and their corresponding labels
def load_dcis_labels(path_to_csv, label_remove="DCIS"):
	data = pandas.read_csv(path_to_csv)

	# remove DCIS cases 
	data = data[data["Match_Hist_ICES"] != label_remove]
	
	train = data[data["Set"] == 1]
	val = data[data["Set"] == 2]
	test = data[data["Set"] == 3]
	
	train_files, val_files, test_files = train["scanID"].tolist(), val["scanID"].tolist(), test["scanID"].tolist()
	train_labels, val_labels, test_labels = train["Match_Recurrence_ICES"].tolist(), val["Match_Recurrence_ICES"].tolist(), test["Match_Recurrence_ICES"].tolist()
	
	# convert problem into 3-class task
	if label_remove == "3-class":
		train_labels = train["Match_Hist_ICES"].astype("category").cat.codes.tolist() 
		val_labels = val["Match_Hist_ICES"].astype("category").cat.codes.tolist() 
		test_labels = test["Match_Hist_ICES"].astype("category").cat.codes.tolist()
		#print(test_labels)
		#print(test["Match_Hist_ICES"])		
	
	return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)
	
	
def get_patient_ids(path_to_csv, list_slideids):
	data = pandas.read_csv(path_to_csv)
	
	#find unique patients with slideids
	patients_found = data.loc[data["scanID"].isin(list_slideids)]
	patient_ids = patients_found.groupby(["CaseID_Path"])["scanID"].unique()
	
	return patient_ids
	
	
	
	
	
