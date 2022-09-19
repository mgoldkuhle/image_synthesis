# split data for k-fold cross-validation patient-wise
# filter unsorted patient .jpg images in a folder per patient (subject) as given in the metadata file
import argparse
import pandas as pd
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default='../data', help='Path for the images to be compared to. Can be directory of images or single image file')
parser.add_argument('--out_path', default='../data/folds', help='Directory for the images to compare. Can be directory of images or single image file')
parser.add_argument('--k', default=5, type=int, help='Number of folds to be used in k-fold cross-validation. The data will be divided into k folds')
parser.add_argument('--syndromes', default=[0, 1, 2, 3, 12], type=list, help='Select which syndromes your data covers as integer indices in the metadata')
args = parser.parse_args()

if __name__ == '__main__':

    data_dir = args.in_path

    # metadata that contains image_ids, subject_ids and syndrome labels
    v0 = pd.read_csv(f"{data_dir}/v1_0_2/metadata/gmdb_train_images_v1.0.2.csv")
    v1 = pd.read_csv(f"{data_dir}/v1_0_2/metadata/gmdb_val_images_v1.0.2.csv")
    v2 = pd.read_csv(f"{data_dir}/v1_0_2/metadata/gmdb_test_images_v1.0.2.csv")
    ids = pd.concat([v0, v1, v2])
    del v0, v1, v2

    syndrome_list = args.syndromes
    ids = ids[ids['label'].isin(syndrome_list)]

    k = args.k
    folds = list(range(1, k + 1))

    out_path = args.out_path

    for i in syndrome_list:
        syndrome_ids = ids[ids['label'] == i]
        unique_patients = syndrome_ids.drop_duplicates(subset='subject', keep='first')
        unique_patients = unique_patients.drop(['image_id', 'label'], axis=1)
        unique_patients = unique_patients.sample(frac=1)  # shuffle unique subjects
        out_length = len(unique_patients)
        unique_patients['fold'] = [folds * out_length][0][:out_length]  # assign fold 1 to k to each subject
        ids_folds = pd.merge(syndrome_ids, unique_patients, on='subject', how='left')  # merge folds and image_ids by subject

        # copy images to new separate directories by syndrome and fold
        for j in folds:
            fold_path = f'{out_path}/{j}/{i}'
            ids_fold = ids_folds[ids_folds['fold'] == j]
            for ix, row in ids_fold.iterrows():
                img = row['image_id']  # image_id
                img_path = os.path.join(data_dir, f'crops_256/{img}.jpg')
                target_path = os.path.join(fold_path, f'{img}.jpg')
                try:
                    shutil.copy(img_path, target_path)
                except FileNotFoundError:
                    print(f'File {img}.jpg does not exist.')
