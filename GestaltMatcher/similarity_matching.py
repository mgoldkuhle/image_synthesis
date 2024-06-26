## similarity_matching.py
# Runs similarity matching on the LFW splits
# Can also be used to find the ideal threshold

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from lib.models.face_recog_net import FaceRecogNet

saved_model_dir = "saved_models"


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Bone Age Test')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--model-type', default='FaceRecogNet', dest='model_type',
                        help='Model type to use. (Options: FaceRecogNet)')
    parser.add_argument('--act_type', default='ReLU', dest='act_type',
                        help='activation function to use in UNet. (Options: ReLU, PReLU, LeakyReLU, Swish)')
    parser.add_argument('--in_channels', default=1, dest='in_channels')
    parser.add_argument('--num_classes', default=10575, dest='num_classes')

    # File specific
    parser.add_argument('--similarity', default='Cosine', dest='sim_type',
                        help='Type of similarity scoring/matching to use')
    parser.add_argument('--dataset', default='LFW', dest='dataset',
                        help='Dataset to use to match similarity on')
    parser.add_argument('--is_fitting', action='store_true', dest='is_fitting',
                        help='When True will try and learn/decide the similarity threshold ... '
                             '(if assigned on the fitting split)')

    return parser.parse_args()


def fit(model, dataset, sim_type, device):
    if sim_type == 'Cosine':
        sim_f = nn.CosineSimilarity()
    else:
        print(f"Unknown sim_type ({sim_type} given), exiting ...")
        exit()

    if dataset == 'LFW':
        dataset_dir = '../data/LFW/'
        imgs_dir = f"{dataset_dir}lfw_cropped/"

        def to_filenames(df, same_diff):
            def to_filename(name, number, prefix_dir='', postfix_img='_crop_square'):
                if prefix_dir != '':
                    prefix_dir += '\\'
                return f"{prefix_dir}{name}\\{name}_{number:04}{postfix_img}.jpg"

            return to_filename(df[0], int(df[1])), \
                   to_filename(df[0] if same_diff == 'same' else df[2],
                               int(df[2]) if same_diff == 'same' else int(df[3]))

        # fit on view1
        pairs_file = '../data/LFW/pairs_view1.csv'
        pairs = pd.read_csv(pairs_file, names=['name', 'img1', 'img2', 'img3'], skiprows=1)
        same_diff = ['same', 'diff']
        same_diff_sim_scores = [[], []]

        for pair in pairs.values:
            filename1, filename2 = to_filenames(pair, 'same' if np.isnan(pair[3]) else 'diff')
            try:
                img1 = TF.to_tensor(TF.to_grayscale(Image.open(f"{imgs_dir}{filename1}"))).unsqueeze(0).to(device)
                img2 = TF.to_tensor(TF.to_grayscale(Image.open(f"{imgs_dir}{filename2}"))).unsqueeze(0).to(device)
            except FileNotFoundError:
                # print("Missing file (likely due to pruning), skipping ...")
                continue

            with torch.no_grad():
                _, img1_rep = model(img1)
                _, img2_rep = model(img2)
                sim_score = sim_f(img1_rep, img2_rep)
            # print(sim_score.item())
            same_diff_sim_scores[0 if np.isnan(pair[3]) else 1].append(sim_score.item())

        threshold_scores = []
        for thresh in range(0, 1000):
            thresh /= 1000
            same = [True if ss >= thresh else False for ss in same_diff_sim_scores[0]]
            diff = [True if ss < thresh else False for ss in same_diff_sim_scores[1]]
            threshold_scores.append(sum(same) + sum(diff))

        threshold_scores = np.array(threshold_scores)
        best_threshold = np.argmax(threshold_scores) / 1000
        best_acc = np.max(threshold_scores) / (len(same_diff_sim_scores[0]) + len(same_diff_sim_scores[1]))
        print(f"Threshold {best_threshold} reached the highest accuracy of {best_acc}")

    return best_threshold


def test(model, dataset, sim_type, threshold, device):
    if sim_type == 'Cosine':
        sim_f = nn.CosineSimilarity()
    else:
        print(f"Unknown sim_type ({sim_type} given), exiting ...")
        exit()

    if dataset == 'LFW':
        dataset_dir = '../data/LFW/'
        imgs_dir = f"{dataset_dir}lfw_cropped/"

        def to_filenames(df, same_diff):
            def to_filename(name, number, prefix_dir='', postfix_img='_crop_square'):
                if prefix_dir != '':
                    prefix_dir += '\\'
                return f"{prefix_dir}{name}\\{name}_{number:04}{postfix_img}.jpg"

            return to_filename(df[0], int(df[1])), \
                   to_filename(df[0] if same_diff == 'same' else df[2],
                               int(df[2]) if same_diff == 'same' else int(df[3]))

        accs = []
        for i in range(0, 10):
            # test on split of view2
            pairs_file = f"../data/LFW/test_splits/view2_split_{i}.csv"
            pairs = pd.read_csv(pairs_file, names=['name', 'img1', 'img2', 'img3'], delimiter='\t')
            same_diff_sim_scores = [[], []]

            for pair in pairs.values:
                filename1, filename2 = to_filenames(pair, 'same' if np.isnan(pair[3]) else 'diff')
                try:
                    img1 = TF.to_tensor(TF.to_grayscale(Image.open(f"{imgs_dir}{filename1}"))).unsqueeze(0).to(device)
                    img2 = TF.to_tensor(TF.to_grayscale(Image.open(f"{imgs_dir}{filename2}"))).unsqueeze(0).to(device)
                except FileNotFoundError:
                    # print("Missing file (likely due to pruning), skipping ...")
                    continue

                with torch.no_grad():
                    _, img1_rep = model(img1)
                    _, img2_rep = model(img2)
                    sim_score = sim_f(img1_rep, img2_rep)
                # print(sim_score.item())
                same_diff_sim_scores[0 if np.isnan(pair[3]) else 1].append(sim_score.item())

            same = [True if ss >= threshold else False for ss in same_diff_sim_scores[0]]
            diff = [True if ss < threshold else False for ss in same_diff_sim_scores[1]]
            threshold_scores = (sum(same) + sum(diff))

            acc = threshold_scores / (len(same_diff_sim_scores[0]) + len(same_diff_sim_scores[1]))
            accs.append(acc)
            print(f"\tThreshold {threshold} had an accuracy of {acc} on split {i}")
        accs = np.array(accs)
        print(f"Threshold {threshold} had a mean accuracy of {accs.mean()} with an std of {accs.std()}")


def main():
    # Training settings
    args = parse_args()

    print("Running similarity matching on LFW splits.")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.act_type == "ReLU":
        act_type = nn.ReLU
    elif args.act_type == "PReLU":
        act_type = nn.PReLU
    elif args.act_type == "LeakyReLU":
        act_type = nn.LeakyReLU
    else:
        print(f"Invalid ACT_type given! (Got {args.act_type})")
        act_type = nn.ReLU

    if args.model_type == 'FaceRecogNet':
        model = FaceRecogNet(in_channels=args.in_channels, num_classes=args.num_classes, act_type=act_type).to(device)
        model_name = "FaceRecogNet"

        model.load_state_dict(
            torch.load(f"saved_models/s7_casia_adam_FaceRecogNet_e50_ReLU_BN_bs100.pt",
                       map_location=device))
    else:
        print(f"No valid model type given! (got model_type: {args.model_type})")
        exit(0)

    # Set to evaluation mode, we're no longer training
    model.eval()

    threshold = -1
    if args.is_fitting:
        threshold = fit(model, args.dataset, args.sim_type, device=device)
    # else:
    test(model, args.dataset, args.sim_type, threshold=(0.327 if threshold == -1 else threshold), device=device)


if __name__ == '__main__':
    main()
