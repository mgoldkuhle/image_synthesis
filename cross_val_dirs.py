# copy folded patient .jpg images created by patient_split.py in a directory structure to be used in stylegan2 training with cross-validation
# the images that belong to fold x are treated as test images while the images belonging to all other folds are treated as training images
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default='../data/folds', help='Path for the images folds')
parser.add_argument('--out_path', default='../data/cross_validation', help='Path for the training and test set folders')
parser.add_argument('--syndromes', default=[0, 1, 2, 3, 12], type=list, help='Select which syndromes your data covers as integer indices in the metadata')
parser.add_argument('--verbose', default=True, type=bool, help='Print progress')
args = parser.parse_args()

verbose = args.verbose
syndromes = args.syndromes
in_path = args.in_path
out_path = args.out_path
folds = range(1, 6)

file_names = os.listdir(in_path)
file_names = [name for name in file_names if '.jpg' in name or '.png' in name]

for fold in folds:
    others = [x for x in folds if x != fold]
    fold_path = os.path.join(in_path, str(fold))
    for syndrome in syndromes:
        syndrome_path = os.path.join(fold_path, str(syndrome))
        file_names = os.listdir(syndrome_path)
        file_names = [name for name in file_names if '.jpg' in name or '.png' in name]

        for file_name in file_names:
            image_path = os.path.join(syndrome_path, file_name)
            target_path = os.path.join(out_path, f'{fold}/test/{syndrome}')
            shutil.copy(image_path, target_path)
            if verbose:
                print(f'Copying test images for fold {fold}')
                print(f'Copied from {image_path} to {target_path}')

        for other in others:
            for file_name in file_names:
                image_path = os.path.join(syndrome_path, file_name)
                target_path = os.path.join(out_path, f'{other}/train/{syndrome}')
                shutil.copy(image_path, target_path)
                print(f'Copying train images for fold {fold}')
                print(f'Copied from {image_path} to {target_path}')
