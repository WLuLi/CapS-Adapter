# Script to encode the SuS-SD images using CLIP's image encoders

import os
import numpy as np
from torchvision.datasets import DatasetFolder
import torch
import clip
import torch.nn.functional as F
import random
from tqdm import tqdm
from PIL import Image
import argparse
import logging

FEATURE_PATH = "./data/features/"
CLIP_MODEL_PATH = './models'
DATASET_PATH = './data/caps/sus-sd/{}/{}'

logging.basicConfig(filename='generate_sus_sd.log', filemode='a', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class SelectedImagesDataset(DatasetFolder):
    def __init__(self, root, num_images, transform=None):
        super().__init__(root, loader=pil_loader, extensions=('jpg', 'jpeg', 'png', 'bmp', 'gif'), transform=transform)
        self.num_images = num_images
        self.samples = self._select_samples()

    def _select_samples(self):
        selected_samples = []
        for class_index, class_name in enumerate(self.classes):
            class_samples = [s for s in self.samples if s[1] == class_index]
            selected_samples.extend(random.sample(class_samples, min(self.num_images, len(class_samples))))
        return selected_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# feature dimensions for each model
feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

parser = argparse.ArgumentParser()
# number of augmentations to apply for averaging visual features
parser.add_argument('--augment_epoch', type=int, default=10)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--prompt_shorthand', type=str, default='photo')
parser.add_argument('--regenerate', action='store_true', help='Regenerate features')
parser.add_argument('--images_per_class', type=int, default=10, help='Number of images per class to encode')
args = parser.parse_args()

# dummy parameters for dataloader
args.k_shot = 2
args.val_batch_size = 64 
args.train_batch_size = 256 

random.seed(1)
torch.manual_seed(1)

req_models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
dataset = args.dataset

for model_name in req_models:
    name = model_name
    print('Current model: '+str(name))

    disp_name = name
    if('/' in name):
        disp_name = name.replace('/', '')

    model, preprocess = clip.load(name, download_root=CLIP_MODEL_PATH)
    model.eval()

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    print('Processing current dataset: '+dataset)

    features_path = FEATURE_PATH + args.dataset

    if not os.path.exists(features_path):
        os.makedirs(features_path)
    store_features_path = features_path + "/sus_sd_{}_{}_f_m{}.pt".format(args.prompt_shorthand, dataset, disp_name)
    store_targets_path = features_path + "/sus_sd_{}_{}_t_m{}.pt".format(args.prompt_shorthand, dataset, disp_name)

    if args.regenerate:
        if os.path.exists(store_features_path):
            os.remove(store_features_path)
        if os.path.exists(store_targets_path):
            os.remove(store_targets_path)
        load_stoch = False
    else:
        load_stoch = os.path.exists(store_features_path) and os.path.exists(store_targets_path)

    dataset_path = DATASET_PATH.format(args.dataset, args.prompt_shorthand)

    sd_images = SelectedImagesDataset(dataset_path, args.images_per_class, transform=preprocess)

    dataloader = torch.utils.data.DataLoader(sd_images, batch_size=args.val_batch_size, num_workers=8, shuffle=False)        

    # ------------------------------------------saving features------------------------------------------
    print('start saving sus sd image features')

    if not load_stoch:

        train_images_targets = []
        train_images_features_agg = []

        # take average of features over multiple augmentations for a more robust feature set
        with torch.no_grad():
            for augment_idx in range(args.augment_epoch):
                train_images_features = []

                print('Augment time: {:} / {:}'.format(augment_idx, args.augment_epoch))
                for i, (images, target) in enumerate(tqdm(dataloader)):
                    images = images.cuda()
                    image_features = model.encode_image(images)
                    train_images_features.append(image_features)

                    if augment_idx == 0:
                        target = target.cuda()
                        train_images_targets.append(target)

                images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
                train_images_features_agg.append(images_features_cat)
            
        # concatenate and take mean of features from multiple augment runs
        train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(dim=0)
        # L2 normalise image embeddings from few shot dataset -- dim NKxC
        train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
        # dim CxNK
        train_images_features_agg = train_images_features_agg.permute(1, 0)

        # convert all image labels to one hot labels -- dim NKxN
        train_images_targets = F.one_hot(torch.cat(train_images_targets, dim=0)).half()

        assert train_images_features_agg.shape[0]==feat_dims[name], 'train_images_features_agg is not of shape CxNK'

        print('Storing features to: '+store_features_path+' and '+store_targets_path)
        logging.info('Storing features to: '+store_features_path+' and '+store_targets_path + ' images_per_class: '+str(args.images_per_class))

        # dim CxNK
        torch.save(train_images_features_agg, store_features_path)
        # dim NKxN
        torch.save(train_images_targets, store_targets_path)

    else:
        print('Loading features from: '+store_features_path+' and '+store_targets_path)
        logging.info('Loading features from: '+store_features_path+' and '+store_targets_path + ' images_per_class: '+str(args.images_per_class))

        # dim CxNK
        train_images_features_agg = torch.load(store_features_path)
        # dim NKxN
        train_images_targets = torch.load(store_targets_path)
