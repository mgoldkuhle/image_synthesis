# ref: https://github.com/JulianPinzaru/stylegan2-ada-pytorch-multiclass-labels/blob/main/make_json_labels.py
# if conditional model shall be trained:
# 1. source folder structure should be like
# source
# --label0
# ----image0.jpg
# ----image1.jpg
# --label1
# ----image2.jpg
# ----image3.jpg
# --label2
# ----image4.jpg
# ----image5.jpg
# --dataset.json
#
# dataset.json contains the following:
# {
#     "labels":
#         [
#             ["label0/image0.jpg", 0], ["label0/image1.jpg", 0],
#             ["label1/image2.jpg", 1], ["label1/image3.jpg", 1],
#             ["label2/image4.jpg", 2], ["label2/image5.jpg", 2],
#         ]
# }
# to generate this file:
# python make_json_labels.py --input_folder path --output_folder path
# check that the file strings are correct "label/image.jpg" and NOT in os.path format "label\\image.jpg"
# compress all dirs and inside(!) source - not source itself - into .zip file and use it as --source for dataset_tool.py

import argparse
import os
import json


def parse_args():
	desc = "Tool to create multiclass json labels file for stylegan2-ada-pytorch" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--verbose', action='store_true',
		help='Print progress to console.')

	parser.add_argument('--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	args = parser.parse_args()
	return args


def main():
	global args
	args = parse_args()
	base_dir = ''

	remakePath = args.output_folder
	if not os.path.exists(remakePath):
		os.makedirs(remakePath)

	data_dict = {'labels': []}
	label_counter = 0

	with open(os.path.join(remakePath, 'dataset.json'), 'w') as outfile:

		for root, subdirs, files in os.walk(args.input_folder):
			if len(subdirs) > 0:
				base_dir = root	
				continue

			current_subdir = os.path.split(root)[1]

			for filename in files:
			#	file_path = os.path.join(current_subdir, filename) # this original line doesn't work with dataset_tool
				file_path = (current_subdir + '/' + filename)
				if(args.verbose): print('\t- file %s (full path: %s)' % (filename, file_path))
				data_dict['labels'].append([file_path, label_counter])
				
			label_counter += 1

		json.dump(data_dict, outfile)


if __name__ == "__main__":
	main()

