'''
File: 	test.py
Author: 	Shazia Akbar (sakbar@sri.utoronto.ca)

Description:
Test file for extracting patches from a directory containing multiple svs files. 
All patches are stored in the same location and no assumptions are made about 
the location of extracted patches, therefore a dense extraction is performed 
spaced (512, 512) apart.
'''

import extract
import glob

directory = "/labs/jaylabs/amartel_data2/histology/CCSRI_Project/PostTreatment/PostTreatment_25Slides_14Cases"
save_location = "/home/sakbar/extract/out/"

list_files = glob.glob(directory + "/*.svs")

for svsfile in list_files:
	print("Extracting " + svsfile)	
	einst = extract.TissueLocator(svsfile, tile_size = (512, 512), mode="random", num_tiles_per_slide=102)
	einst.extract_patches_and_save(out_location = save_location, workers=4)

print("Completed extraction.")
