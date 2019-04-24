import os
import numpy as np
import utils
import pandas
from openslide import open_slide
from sklearn.metrics import roc_curve, auc, brier_score_loss, classification_report
from skimage.morphology import binary_dilation, disk
from scipy.misc import imread, toimage
import matplotlib.pyplot as plt
from itertools import cycle

from keras.models import load_model

PATH_TO_SLIDES = '/labs3/amartel_data3/histology/Data/DCIS_cohort/DCIS_Training_Dataset/'
PATH_TO_MASKS = '/labs3/amartel_data3/histology/Data/DCIS_cohort/DCIS_Training_Dataset/dcis_seg/'
PATIENT_INFO = '/labs3/amartel_data3/histology/Data/DCIS_cohort/DCIS_Training_Dataset/CaseSetsDefined.csv'

class Validate():
	
	def __init__(self, model, slide_analysis = False, patient_analysis = False, batch_size = 1000, resolution = 1, show_overlay = False, recurr_ignore="NONE"):
		
		self.bool_slide = slide_analysis
		self.bool_patient = patient_analysis
		self.batch_size = batch_size
		self.model = load_model(model)
		self.model_filename = model
		self.pyramid_level = resolution
		self.patch_size = (256, 256)
		self.step = (256, 256)
		self.show_overlay = show_overlay
		self.remove_data = recurr_ignore
		self.regions_to_analyse = ["peri", "duct", "peri-duct"]
		
		# Load test set directory
		_, _, test = utils.load_dcis_labels(PATIENT_INFO, label_remove=recurr_ignore)
		self.test_set = [os.path.join(PATH_TO_SLIDES, str(int(x))) + '.svs' for x in test[0]]
		self.gt = test[1]
		self.test_slides = test[0]
		
		# Create directory to save visualizations and results if it doesn't exist
		self.save_directory = os.path.split(model)[0] + "/analysis_" + recurr_ignore + "/" 
		if not os.path.exists(self.save_directory):
			os.makedirs(self.save_directory)
		
	def get_mask(self, filename, mask_type, probability_mask=False):
		"""
		Define parts of the image that are to be analysed. If no parameter specified then the entire image will be used.
		If probability_mask is True, then the original probability map is re-thresholded
		"""
		mask = None
		if mask_type is None:
			dims = utils.get_image_dims(_file, resolution=self.pyramid_level)
			mask = np.ones(dims)
		else:
			name = filename.split('/')[-1]
			
			if probability_mask is True:
				mask_filename = PATH_TO_MASKS + name[:-4] + "_prob.jpg"
				mask = imread(mask_filename).astype('float32')
				mask = (mask > 0.35).astype('bool')
			else:
				mask_filename = PATH_TO_MASKS + name[:-4] + "_seg.jpg"
				mask = imread(mask_filename).astype('bool')
			
			# strip corners
			mask[:50, :] = 0
			mask[:, :50] = 0
			mask[-50:, :] = 0
			mask[:, -50:] = 0
			# todo: process using Nikhil's method and save offline
			
			dilated_mask = binary_dilation(mask, disk(60))
			if mask_type == "peri":
				mask = np.bitwise_xor(dilated_mask, mask)
			elif mask_type == "peri-duct":
				mask = dilated_mask
			elif mask_type == 'duct':
				mask = mask
			
		return mask
		
	def get_model_predictions(self, model, filename, locations):
		"""
		Basic function for retrieving recurrence outcomes for every patch in mask (above)
		"""
		predictions = np.array([])
		for j in range(0, len(locations), self.batch_size):
			patches = utils.get_patch_from_locations(filename, locations[j:min(j+self.batch_size, len(locations))], self.patch_size, resolution_level=self.pyramid_level)
			_p = model.predict(patches)
			
			# support from multi-class output
			if _p.shape[1] > 2:
				if len(predictions) is 0:
					predictions = _p
				else:
					predictions = np.concatenate((predictions, _p), axis=0)
			else:	
				predictions = np.concatenate((predictions, _p[:, 1]), axis=0)
			
		return predictions
	
	def combine_patient_level(self, p):
		"""
		Function for combining predictions at the patient-level
		"""
		
		# identify patients
		patient_ids = utils.get_patient_ids(PATIENT_INFO, self.test_slides)
		
		# collate predictions at patient-level
		patient_predictions, new_gt = [], []
		for ids in patient_ids:
			this_p = []
			for slide in ids:
				this_index = np.where(self.test_slides == slide)[0][0]
				this_p.append(p[this_index])
				this_gt = self.gt[this_index]

			new_gt.append(this_gt)
			patient_predictions.append(self.combine_slide_level(this_p))
			
		patient_predictions = np.array(patient_predictions).squeeze()
		new_gt = np.array(new_gt).squeeze()
		
		print("Number of patients: {}".format(len(patient_predictions)))	
		#print(patient_predictions, new_gt)

		return patient_predictions, new_gt
			
	def combine_slide_level(self, p, combine_method='average'):
		"""
		Function for combining predictions extracted from each slide
		"""
		if combine_method is "majority":
			# histogram predictions
			
			def get_max_thresholds(a):
				h, thresholds  =  np.histogram(p, 1000, range=(0.,1.))
				index_max = np.argmax(h)													
				return (thresholds[index_max - 1] + thresholds[index_max]) / 2
			
			pred_value = np.apply_along_axis(lambda a: get_max_thresholds(a), 0, p)
			
		elif combine_method is "average":
			pred_value = np.sum(p, axis=0) / len(p)
		else:
			print('Not implemented yet.')
		
		pred_value = pred_value[np.newaxis, ...]
				
		return pred_value	
		
	def perform_roc_analysis(self, predictions, ground_truth, result_filename):
		
		# different outputs depending on if two-class or three-class predictions
		if len(predictions.shape) > 2:
			
			f = open(self.save_directory + result_filename + ".txt", "w")
			
			lw=2
			colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
			plt.figure()
			for i, color in zip(range(predictions.shape[1]), colors):
				fpr, tpr, roc_thresholds = roc_curve((ground_truth == i).astype('int'), predictions[:, i])
				plt.plot(fpr, tpr, color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, auc(fpr, tpr)))
			
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.0])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.legend(loc="lower right")
			plt.savefig(self.save_directory + result_filename + "_roc.png")
			plt.close()
			
			f.write("roc auc class {}: {}\n".format(i, auc(fpr, tpr)))
			f.write("brier score class {}: {}\n\n".format(i, brier_score_loss((ground_truth == i).astype('int'), predictions[:, i])))
			
			f.write("classification report:\n {}\n".format(classification_report(ground_truth, np.argmax(predictions, axis=1).astype('int'), digits = 4)))
				
			f.close()
			
		else: 
			fpr, tpr, roc_thresholds = roc_curve(ground_truth, predictions)
			
			plt.figure()
			plt.plot(fpr, tpr)
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.0])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.savefig(self.save_directory + result_filename + "_roc.png")

			f = open(self.save_directory + result_filename + ".txt", "w")
			f.write("roc auc: {}\n\n".format(auc(fpr, tpr)))
			
			f.write("brier score: {}\n\n".format(brier_score_loss(ground_truth, predictions)))
			
			f.write("classification report:\n {}\n\n".format(classification_report(ground_truth, (predictions > 0.5).astype('int'), digits = 4)))
			
			'''
			f.write("\n fpr: \n")
			np.savetxt(f, fpr, delimiter=',')
			f.write("\n tpr: \n")
			np.savetxt(f, tpr, delimiter=',')
			f.write("\n roc thresholds: \n")
			np.savetxt(f, roc_thresholds, delimiter=',')
			'''

			f.close()
		return
		
	def save_prediction_map(self, filename, locations, predictions, label):
		name = filename.split('/')[-1]
		
		# lookup original size of digital slide
		_slide = open_slide(filename)
		orig_dims = _slide.level_dimensions[self.pyramid_level]
		level0_dims = _slide.level_dimensions[0]
		_slide.close()
		
		# create empty overlay
		overlay = np.zeros((int(np.ceil(orig_dims[1] / self.step[0])), int(np.ceil(orig_dims[0] / self.step[1]))), dtype='float32')
		
		# assign predictions to correct locations
		for j, loc in enumerate(locations):
			if predictions.shape[1] > 2:
				overlay[int(loc[1] / self.step[0]), int(loc[0] / self.step[1])] = predictions[j, 1]
			else:
				overlay[int(loc[1] / self.step[0]), int(loc[0] / self.step[1])] = predictions[j]
			
		toimage(overlay, cmin=0.0, cmax=1.0).save(self.save_directory + name[:-4] + "_" + str(label) + ".jpg")
		
		'''
		plt.figure()
		plt.imshow(overlay, vmin=0, vmax=1, cmap=plt.get_cmap('jet'))
		plt.axis('off')
		plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
		plt.savefig(self.save_directory + name[:-4] + "_" + str(label) + ".png", bbox_inches='tight')
		plt.close()
		'''
	
	def get_time_to_event_info(self):
		# load recurrence information and time to recurrence - identical to utils.load_dcis_labels
		patient_data = pandas.read_csv(PATIENT_INFO)
		patient_data = patient_data[patient_data["Match_Hist_ICES"] != self.remove_data]
		patient_data = patient_data[patient_data["Set"] == 3]
		
		# concatenate prediction, labels, time
		analysis_data_outcome = patient_data["Match_Recurrence_ICES"]
		analysis_data_time_to_event = patient_data["Match_Time_Year_ICES"]
		return analysis_data_time_to_event
			
	def run_validation(self):
		"""
		Main function
		"""
		pred_all_peri = pred_all_periduct = pred_all_duct = gt_all_peri = gt_all_periduct = gt_all_duct = np.array([])
		
		print("Number of WSIs: {}".format(len(self.test_set)))
		patient_ids = utils.get_patient_ids(PATIENT_INFO, self.test_slides)
		
		for idx, test_filename in enumerate(self.test_set):
			print(test_filename)
			
			for j in self.regions_to_analyse:
				
				mask = self.get_mask(test_filename, j)
				
				# extract locations from mask
				locations = utils.get_tissue_patch_locations_from_tif_custom(test_filename, self.step, mask, resolution_level=self.pyramid_level)
				
				if len(locations) > 0:
					predictions = self.get_model_predictions(self.model, test_filename, locations)
				else:
					print("Re-analyzing probability map...")
					mask = self.get_mask(test_filename, j, probability_mask = True)
					locations = utils.get_tissue_patch_locations_from_tif_custom(test_filename, self.step, mask, resolution_level=self.pyramid_level)
					predictions = self.get_model_predictions(self.model, test_filename, locations)
				
				# save a visual probability map if requested
				if self.show_overlay:
					self.save_prediction_map(test_filename, locations, predictions, self.gt[idx])

				# convert patch probabilities to single probability per slide if requested	
				if self.bool_slide:
					predictions = self.combine_slide_level(predictions)
					#self.survival_analysis(predictions)
				
				# concatenate results (gt length different for each mask type)
				gt_this = np.tile(self.gt[idx], len(predictions))
				if j == "peri":
					if len(pred_all_peri) is 0:
						pred_all_peri = predictions
						gt_all_peri = gt_this
					else:
						pred_all_peri = np.concatenate([pred_all_peri, predictions])
						gt_all_peri = np.concatenate([gt_all_peri, gt_this])
				elif j == "peri-duct":
					if len(pred_all_periduct) is 0:
						pred_all_periduct = predictions
						gt_all_periduct = gt_this
					else:
						pred_all_periduct = np.concatenate([pred_all_periduct, predictions])
						gt_all_periduct = np.concatenate([gt_all_periduct, gt_this])
				else:
					if len(pred_all_duct) is 0:
						pred_all_duct = predictions
						gt_all_duct = gt_this
					else:
						pred_all_duct = np.concatenate([pred_all_duct, predictions])
						gt_all_duct = np.concatenate([gt_all_duct, gt_this])
		'''				
		pred_all_peri = np.zeros((len(self.test_slides), 3))
		pred_all_peri[:, 2] = 0.9
		gt_all_peri = self.gt
		gt_all_peri[0] = gt_all_peri[1] = gt_all_peri[3] = 1 
		'''	
			
		append  = ""
		if self.bool_slide is True:
			append = "_slide"
			
		if self.bool_patient is True:
			append = "_patient"
			if len(pred_all_peri) > 0:
				pred_all_peri, gt_all_peri = self.combine_patient_level(pred_all_peri)
			if len(pred_all_duct) > 0:
				pred_all_duct, gt_all_duct = self.combine_patient_level(pred_all_duct)
			if len(pred_all_periduct) > 0:
				pred_all_periduct, gt_all_periduct = self.combine_patient_level(pred_all_periduct)
		
		# write results to file	
		if len(pred_all_peri) > 0:
			np.savetxt(self.save_directory + "peri_predictions.npy", pred_all_peri)
			self.perform_roc_analysis(pred_all_peri, gt_all_peri, 'res_peri' + append)
		if len(pred_all_duct) > 0:
			self.perform_roc_analysis(pred_all_duct, gt_all_duct, 'res_duct' + append)
		if len(pred_all_periduct) > 0:
			# save predictions and ground truth for survival analysis 
			if self.bool_slide is True:
				np.savetxt(self.save_directory + "predictions.txt", pred_all_periduct)
				np.savetxt(self.save_directory + "groundtruth.txt", gt_all_periduct)
				np.savetxt(self.save_directory + "timetoevent.txt", self.get_time_to_event_info())
			
			self.perform_roc_analysis(pred_all_periduct, gt_all_periduct, 'res_periduct' + append)
			

	def visualize_attention_maps(self, epochs, interval, test_image):
		"""
		Method for saving attention maps generated during the training process
		"""
		
		# load details of test image
		slidename = PATH_TO_SLIDES + test_image + ".svs"
		dims = utils.get_image_dims(slidename, resolution_level=self.pyramid_level)	
		test_locations = utils.get_tissue_patch_locations_from_tif_custom(slidename, self.step, None, resolution_level=self.pyramid_level)
		
		modelpath = os.path.dirname(self.model_filename)
		
		for i in range(0, epochs + 1, interval):
			print("Analyzing epoch " + str(i) + "/" + str(epochs) + "...")
			
			# load weights for current epoch
			attention_model = load_model(modelpath + "/weights." + str(max(0, i-1)) + ".h5")
			
			# get predictions for specified image
			predictions = self.get_model_predictions(attention_model, slidename, test_locations)
			
			# reshape, visualize and save attention map
			plt.figure()
			plt.imshow(predictions.reshape((int(dims[1]/self.step[0]), -1)), vmin=0, vmax=1, cmap=plt.get_cmap('jet'))
			plt.axis('off')
			plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
			plt.savefig(self.save_directory + test_image + "_a" + str(i) + ".png", bbox_inches='tight')
			plt.close()


# test code
val = Validate('./models/attention-task_top5_sparse2048_switchtwolossv2_pyr1_dense_batch32_3class_checkpoints/weights.99.h5', slide_analysis=True, patient_analysis = True, recurr_ignore="3-class", show_overlay = False)
val.run_validation()
#val.visualize_attention_maps(100, 5, '103723')	

