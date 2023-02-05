import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--runs_path', default='./results/runs.csv', help='Path for the GAN setup summary csv')
parser.add_argument('--network_path', default='../training/', help='Path to the GAN .pkl snapshots relative to stylegan2 repo')
parser.add_argument('--out_path', default='../../data/cross_validation', help='Directory for the synthetic images relative to stylegan2 repo')
parser.add_argument('--cv', default=1, help='CV repetition')
parser.add_argument('--fold', default=2, help='CV fold to generate data for')
parser.add_argument('--trunc', default=0.7, help='Truncation used in styleGAN2 generate, higher value higher variation')
parser.add_argument('--num_images', default=1000, help='Number of generated images')
args = parser.parse_args()

if __name__ == '__main__':

    # GAN setup summary csv
    runs = pd.read_csv(args.runs_path, index_col=0, dtype={'setup': 'str'})
    runs = runs.dropna()  # keeps only rows for cross_validation models
    cv = args.cv
    fold = args.fold
    runs_cv = runs[(runs['cv_fold'] == fold) & (runs['cv_repeat'] == cv)]

    out_path = args.out_path
    truncation_psi = args.trunc
    network_path = args.network_path
    num_images = args.num_images

    os.chdir('synthesis/stylegan2')

    for ix, row in runs_cv.iterrows():
        network_pkl = os.path.join(network_path, str(row['setup']), 'network-snapshot.pkl')
        outdir = os.path.join(out_path, str(fold), 'synth', str(int(row['syndrome'])))
        os.system(f"python generate.py --outdir {outdir} --trunc {truncation_psi} --seeds 0-{num_images-1} --network {network_pkl}")