# for a given fraction p, split all data in a directory into a training (p) directory and a test (1-p) directory randomly
# output directories have to exit before running the script

import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default='./data', help='Path for the images to be compared to. Can be directory of images or single image file')
parser.add_argument('--frac', default=0.8, type=float, help='Fraction p of data to be kept in training set. 1-p will be test set.')
parser.add_argument('--out_train', default='./data/train', help='Directory for the images to compare. Can be directory of images or single image file')
parser.add_argument('--out_test', default='./data/test', help='Directory for the output files')
args = parser.parse_args()

if __name__ == '__main__':
    in_path = args.in_path
    file_names = os.listdir(in_path)

    # split file list in train and test
    frac = args.frac
    n_frac = round(frac * len(file_names))
    train_files = random.sample(file_names, n_frac)
    test_files = [i for i in file_names if i not in train_files]

    # copy files to new dirs
    out_path_train = args.out_train
    out_path_test = args.out_test

    for file in train_files:
        shutil.copy(os.path.join(in_path, file), f'{out_path_train}/{file}')

    for file in test_files:
        shutil.copy(os.path.join(in_path, file), f'{out_path_test}/{file}')


