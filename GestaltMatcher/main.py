# run with python main.py --dataset_dir_path ../../data/cross_validation/1 --train_dir_path justweightedsynth --target_model_dir saved_models/classifier/cross_validation_1_1_justweightedsynth

import argparse
import datetime
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

from lib.datasets.casia_web_face_dataset import CasiaWebFaceDataset
from lib.datasets.gestalt_matcher_dataset import GestaltMatcherDataset, GestaltMatcherDatasetDir
from lib.datasets.gestalt_matcher_dataset_augment import GestaltMatcherDataset_augment
from lib.models.deep_gestalt import DeepGestalt
from lib.models.face_recog_net import FaceRecogNet

margin = torch.tensor(2.).cuda()
zero = torch.tensor(0.).cuda()


# Function that helps set each work's seed differently (consistently)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Verification loss as described in https://arxiv.org/abs/1411.7923
# Currently unused.
def verification_loss(representation_vectors, class_labels):
    # Calculate the distance matrix between all representation vectors in the representation_vectors :: dists
    dists = torch.cdist(representation_vectors, representation_vectors)

    # Compute class similarity labels :: y
    # not sure how to do this properly ...
    y = torch.stack([class_labels == class_label for class_label in class_labels])

    # Compute each loss, based on the class similarity labels y :: losses
    # simplyfied the loss to L+ instead of L (which is L+ and L-)
    losses = (0.5 * dists * y) + (0.5 * torch.maximum((margin - dists) * ~y, zero) ** 2)

    # Sum over all losses in the vector and divide by batch size :: losses
    final_loss = torch.sum(losses) / class_labels.size(0)

    # Find a way to update the margin to minimize the verification loss ... I would say that m -> 0
    # margin not used currently ...

    return final_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Script to train gestalt classifer')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',  # lr=1e-3
                        help='learning rate (default: 0.005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val-interval', type=int, default=100000,
                        help='how many batches to wait before validation is evaluated (and optimizer is stepped).')
    parser.add_argument('--session', type=int, dest='session',
                        help='Session used to distinguish model tests.')
    parser.add_argument('--model-type', default='DeepGestalt', dest='model_type',
                        help='Model type to use. (Options: \'FaceRecogNet\', \'DeepGestalt\')')
    parser.add_argument('--act_type', default='ReLU', dest='act_type',
                        help='activation function to use in UNet. (Options: ReLU, PReLU, LeakyReLU, Swish)')
    parser.add_argument('--in_channels', default=1, dest='in_channels',
                        help='Number of color channels of the images used as input (default: 1)')
    # ** number of classes **
    parser.add_argument('--num_classes', default=5, dest='num_classes', type=int)  # 139 for gmdb, 10575 for casia
    parser.add_argument('--target_class_labels', default=['0', '1', '2', '3', '12'], dest='target_class_labels',
                        help='Labels of target classes to load in the directory mode')
    parser.add_argument('--target_class_names', default=['CdL', 'WB', 'Kabuki', 'Angelman', 'HPMRS'], dest='target_class_names',
                        help='Names of target classes to load in the directory mode')
    # parser.add_argument('--alpha', default=0.0, dest='alpha') # verification loss unused ..
    parser.add_argument('--dataset', default='gmdb', dest='dataset',
                        help='Which dataset to use. (Options: "casia", "gmdb", "gmdb_aug")')
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use tensorboard for logging')
    parser.add_argument('--dataset_dir_path', dest='data_dir',
                        default='../../data/original',
                        help='path to the dataset\'s directory')
    parser.add_argument('--train_dir_path', dest='train_dir',
                        default='train',
                        help='name of the training data folder')
    parser.add_argument('--img_load_src', type=str, dest='img_load_src', choices=['csv','directory'],
                        default='directory', help='src to obtain the train, test and dev list')
    parser.add_argument('--pretrained_model_dir', type=str, dest='pretrained_model_dir', default='./saved_models',
                        help='Directory of the pre-trained model used for transfer learning')
    parser.add_argument('--target_model_dir', type=str, dest='target_model_dir',
                        default='saved_models/classifier/gestalt_classifier_0_1_2_3_12',
                        help='Directory to save the model snapshots')
    parser.add_argument('--load_pretrained', action='store_true', default=True, dest='load_pretrained',
                        help='Enable to continue training on top of a pretrained model')
    parser.add_argument('--pretrained_model', type=str, default='./saved_models/s1_casia_adam_FaceRecogNet_e50_ReLU_BN_bs100.pt',
                        dest='pretrained_model', help='Enable to continue training on top of a pretrained model')

    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, epochs=-1, val_loader=None, scheduler=None):
    model.train()

    # Time measurements
    tick = datetime.datetime.now()

    # Tensorboard Writer
    if args.use_tensorboard:
        session_name = args.target_model_dir.rpartition('/')[2]
        writer = SummaryWriter(
            comment=f"{session_name}")  # removed 'comment' parameter so the computer name isn't included
    global_step = 0

    if epochs == -1:
        epochs = args.epochs

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.
        for batch_idx, (data, target) in enumerate(train_loader):
            # data dim. Nx1x100x100
            data = data.to(device, dtype=torch.float32)
            # target Nx1
            target = target.to(device, dtype=torch.int64).unsqueeze(1)

            # writer.add_graph(model, data)
            # writer.close()
            # pred NxC pred_rep Nx320
            pred, pred_rep = model(data)
            # Pytorch's implementation of cross_entropy inherently uses log-sum-exp rule to directly deal with logits
            # in place of softmax probabilities
            loss = F.cross_entropy(pred, target.view(-1), weight=args.ce_weights)

            loss.backward()

            # Clipping gradients here, if we get exploding gradients we should revise...
            nn.utils.clip_grad_value_(model.parameters(), 0.1)

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if (batch_idx + 1) % args.log_interval == 0:
                tock = datetime.datetime.now()
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t(Elapsed time {:.1f}s)'.format(
                    tock.strftime("%H:%M:%S"), epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                                                      100. * batch_idx / len(train_loader), loss.item(),
                    (tock - tick).total_seconds()))
                tick = tock

                if args.use_tensorboard:
                    writer.add_scalar('Train/ce_loss', loss.item(), global_step)

            if val_loader:
                if (batch_idx + 1) % args.val_interval == 0:
                    # avg_val_loss, t_acc, t5_acc = validate(model, device, val_loader, args)
                    avg_val_loss, t_acc, cm = validate(model, device, val_loader, args)
                    tick = datetime.datetime.now()

                    if args.use_tensorboard:
                        writer.add_scalar('Val/ce_loss', avg_val_loss, global_step)
                        writer.add_scalar('Val/top_acc', t_acc, global_step)
                        # writer.add_scalar('Val/top_5_acc', t5_acc, global_step)
                        writer.add_figure("Confusion matrix", cm, global_step)

            global_step += args.batch_size

        # Epoch is completed
        print(f"Overall average training loss: {epoch_loss / len(train_loader):.6f}")
        if args.use_tensorboard:
            writer.add_scalar('Train/ce_loss', epoch_loss / len(train_loader), global_step)

        if scheduler:
            scheduler.step()

        # gradually increase alpha each epoch - currently unused
        # args.alpha *= 1.065

        # Save model        
        print(
            f"Saving model in: {args.target_model_dir}/s{args.session}_{args.dataset}_adam_{args.model_type}_e{epoch}"
            f"_{args.act_type}_bs{args.batch_size}.pt")
        if not os.path.exists(args.target_model_dir):
            os.makedirs(args.target_model_dir)
        torch.save(
            model.state_dict(),
            f"{args.target_model_dir}/s{args.session}_{args.dataset}_adam_{args.model_type}_e{epoch}"
            f"_{args.act_type}_bs{args.batch_size}.pt")

        # Plot the performance on the validation set
        # avg_val_loss, t_acc, t5_acc = validate(model, device, val_loader, args)
        avg_val_loss, t_acc, cm = validate(model, device, val_loader, args)

        if args.use_tensorboard:
            writer.add_scalar('Val/ce_loss', avg_val_loss, global_step)
            writer.add_scalar('Val/top_acc', t_acc, global_step)
            # writer.add_scalar('Val/top_5_acc', t5_acc, global_step)
            writer.add_figure("Confusion matrix", cm, global_step)

    if args.use_tensorboard:
        writer.flush()
        writer.close()


def validate(model, device, val_loader, args, out=False):
    model.eval()
    val_ce_loss = 0.
    top_acc = 0.
    # top_5_acc = 0.

    tick = datetime.datetime.now()
    with torch.no_grad():
        diag = torch.eye(args.val_bs, device=device)
        nb_classes = args.num_classes
        confusion_matrix = np.zeros((nb_classes, nb_classes))

        for idx, (data, target) in enumerate(val_loader):
            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.int64).unsqueeze(1)

            pred, pred_rep = model(data)
            val_ce_loss += F.cross_entropy(pred, target.view(-1), weight=args.ce_weights).item()

            if out:
                for i in range(args.val_bs):
                    print(f"{target[i].item()},{pred[i].tolist()}")

            # extra stats
            # use only top-1 accuracy for gestalt_classifier
            max_pred, max_idx = torch.max(pred, dim=-1)
            top_acc += torch.sum((target == max_idx) * diag).item()

            # confusion matrix
            for t, p in zip(target.view(-1), max_idx.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # top_pred, top_idx = torch.topk(pred, k=5, dim=-1)
            # top_5_acc += np.sum([target[i] in top_idx[i] for i in range(args.val_bs)]).item()  # ... yep, quite ugly

    top_acc = torch.true_divide(top_acc, len(val_loader) * args.val_bs).item()
    # top_5_acc = torch.true_divide(top_5_acc, len(val_loader) * args.val_bs).item()

    # confusion matrix
    plt.figure(figsize=(10, 7))
    class_names = args.target_class_names
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, cmap="crest", annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_ = heatmap.get_figure()
    plt.close()

    model.train()

    print(f"Average BCE Loss ({val_ce_loss / len(val_loader)}) during validation")
    # print(f"\tAverage accuracy: {top_acc}, top-5 accuracy: {top_5_acc}")
    print(f"\tAverage accuracy: {top_acc}")
    print(f"Elapsed time during validation: {(datetime.datetime.now() - tick).total_seconds():.1f}s")

    # return val_ce_loss / len(val_loader), top_acc, top_5_acc
    return val_ce_loss / len(val_loader), top_acc, fig_


def main():
    # Training settings
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"Using {'GPU.' if use_cuda else 'CPU, as was explicitly requested, or as GPU is not available.'}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # torch.set_deterministic(True)

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    kwargs = {}
    if use_cuda:
        # Preprocessing is quick enough to use only 4 workers without wasting time much time (maybe 0.1sec per 1000 batches)
        kwargs.update({'num_workers': 4, 'pin_memory': True})

    dataset_train = dataset_val = None
    lookup_table = None
    if args.dataset == 'casia':
        dataset = CasiaWebFaceDataset(in_channels=args.in_channels,
                                      imgs_dir="../data/CASIA-cropped-pruned/")

    if args.img_load_src == 'csv':

        if args.dataset == "gmdb":
            dataset_train = GestaltMatcherDataset(
                in_channels=args.in_channels, img_postfix='',
                imgs_dir=os.path.join(args.data_dir, 'images_cropped/'),
                target_file_path= os.path.join(args.data_dir, "gmdb_metadata/gmdb_val_images_v1.csv"))
            dataset_val = GestaltMatcherDataset(
                in_channels=args.in_channels, img_postfix='', augment=False,
                imgs_dir=os.path.join(args.data_dir, 'images_cropped/'),
                target_file_path= os.path.join(args.data_dir, "gmdb_metadata/gmdb_val_images_v1.csv"),
                lookup_table=dataset_train.get_lookup_table())

        elif args.dataset == "gmdb_aug":
            dataset_train = GestaltMatcherDataset_augment(
                in_channels=args.in_channels, img_postfix='_rot',
                imgs_dir=os.path.join(args.data_dir, 'images_rot/'),
                target_file_path=os.path.join(args.data_dir, "gmdb_metadata/gmdb_train_images_v1.csv"))

            dataset_val = GestaltMatcherDataset_augment(
                in_channels=args.in_channels, img_postfix='_rot', augment=False,
                imgs_dir=os.path.join(args.data_dir, 'images_rot/'),
                target_file_path=os.path.join(args.data_dir, "gmdb_metadata/gmdb_val_images_v1.csv"),
                lookup_table=dataset_train.get_lookup_table())

        dist = dataset_train.get_distribution()
        print(f"Training dataset size: {sum(dist)}, with {len(dist)} classes and distribution: {dist}")
        print(f"Validation dataset size: {sum(dataset_val.get_distribution())}, "
              f"with distribution: {dataset_val.get_distribution()}")
        lookup_table = dataset_train.get_lookup_table()
        if lookup_table is not None:
            f = open("lookup_table.txt", "w+")
            f.write("index_id to disorder_id\n")
            f.write(f"{lookup_table}")
            f.flush()
            f.close()

        # No split pre-defined, lets randomly make one .. (used for e.g. CASIA)
        if dataset_train == None:
            # Use 10.25% of the dataset for validation
            n_val = int(len(dataset) * 0.1025)
            n_train = len(dataset) - n_val
            dataset_train, dataset_val = random_split(dataset, [n_train, n_val])

# reference for directory based data loading: https://github.com/AshAswin/Explaining-GestaltMatcher

    elif args.img_load_src == 'directory':
        if args.dataset == "gmdb":
            dataset_train = GestaltMatcherDatasetDir(split_dir=os.path.join(args.data_dir, args.train_dir),
                                                     in_channels=args.in_channels, img_postfix='', augment=True,
                                                     img_extension='.jpg', target_labels=args.target_class_labels)

            dataset_val = GestaltMatcherDatasetDir(split_dir=os.path.join(args.data_dir, 'val'),
                                                   in_channels=args.in_channels, img_postfix='', augment=False,
                                                   img_extension='.jpg', target_labels=args.target_class_labels)

        elif args.dataset == "gmdb_aug":
            pass

    # See if we can get the distribution of the dataset for weighted cross entropy
    try:
        dist = dataset_train.get_distribution()
        print(f"Training dataset distribution: {dist}")
    except AttributeError:
        dist = None

    # Set the batch size of the validation set loader to as high as possible (max = args.batch_size)
    args.val_bs = len(dataset_val) if len(dataset_val) < args.batch_size else args.batch_size
    
    train_loader = torch.utils.data.DataLoader(dataset_train, **kwargs, shuffle=True, batch_size=args.batch_size,
                                               worker_init_fn=seed_worker)
    val_loader = torch.utils.data.DataLoader(dataset_val, pin_memory=True, num_workers=0, shuffle=False, drop_last=True,
                                             worker_init_fn=seed_worker,
                                             batch_size=args.val_bs)

    # Attempt to deal with data imbalance: inverse frequency divided by lowest frequency class (min: 0.5, max: 1.0)
    if dist is not None:
        args.ce_weights = (torch.tensor([(sum(dist) / freq) / (sum(dist) / min(dist)) for freq in dist]).float()
                           .to(device)) * 0.5 + 0.5
    else:
        args.ce_weights = None
    print(f"Weighted cross entropy weights: {args.ce_weights}")

    if args.act_type == "ReLU":
        act_type = nn.ReLU
    elif args.act_type == "PReLU":
        act_type = nn.PReLU
    elif args.act_type == "LeakyReLU":
        act_type = nn.LeakyReLU
    else:
        raise NotImplementedError

    if args.model_type == 'FaceRecogNet':
        model = FaceRecogNet(in_channels=args.in_channels, num_classes=args.num_classes, act_type=act_type).to(device)
    elif args.model_type == 'DeepGestalt':
        model = DeepGestalt(in_channels=args.in_channels, num_classes=args.num_classes, act_type=act_type,
                            freeze=False, pretrained=args.load_pretrained, pretrained_path=args.pretrained_model,
                            device=device).to(device)
    else:
        print(f"No valid model type given! (got model_type: {args.model_type})")
        raise NotImplementedError
    
    # Set log intervals
    args.log_interval = args.log_interval // args.batch_size
    args.val_interval = args.val_interval // args.batch_size

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
    # optimizer = optim.Adam([
    #     {'params': model.base.parameters()},
    #     {'params': model.classifier.parameters(), 'weight_decay': 5e-4}
    # ], lr=args.lr, weight_decay=0.)
    scheduler = None
    
    # Call explicit model weight initialization
    model.init_layer_weights()
    
    # Continue training/testing:
    # Load pretrained CASIA model
    # model.load_state_dict(torch.load(f"saved_models/<saved weights>.pt", map_location=device))
    
    train(args, model, device, train_loader, optimizer, val_loader=val_loader, scheduler=scheduler)
    validate(model, device, val_loader, args, out=True)


if __name__ == '__main__':
    main()
