# image_synthesis
Generating portrait images with genetically dysmorphic features

cross_val_dirs.py to copy split patient .jpg images created by patient_split.py in a directory structure to be used in stylegan2 training

face_similarities.py recognizes portrait images that share the same identity

image_comparison.py to highlights the pixel-wise differences between two images

patient_split.py filters unsorted patient .jpg images in a folder per patient as given in the metadata file

syndrome_filter.py extracts images from a directory that belong to a certain syndrome as indicated by a metadata file

rename_files.py renames files in a directory with an ascending trailing index

make_json_labels.py automatically creates the .json label file needed for conditional training of styleGAN2-ada

see stylegan2_prerequisites.txt on the prerequisites to run stylegan2-ada-pytorch on a local windows machine or a linux server
