# Script to encode the CapS images using CLIP's image encoders

import os
import numpy as np
from torchvision.datasets import DatasetFolder
import torch
import clip
import json
import torch.nn.functional as F
import random
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

CLIP_MODEL_PATH = './models'
FEATURE_PATH = './data/features/{}/{}'
STORE_FEATURE_DIR = '/caps_{}_f_m{}.pt'
STORE_CAPTION_FEATURE_DIR = '/caps_{}_c_m{}.pt'
STORE_TARGET_DIR = '/caps_{}_t_m{}.pt'
CAPS_PATH = './data/caps/caps_sd/{}/{}'

# feature dimensions for each model
feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

# models to encode images with
req_models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']

random.seed(1)
torch.manual_seed(1)

def parse_args():
    parser = argparse.ArgumentParser()
    # number of augmentations to apply for averaging visual features
    parser.add_argument('--augment_epoch', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--images_per_class', type=int, default=10, help='Number of images per class to encode')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate features')
    parser.add_argument('--prompt_length', type=int, default=20)
    args = parser.parse_args()
    return args

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_captions(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    captions = {item['id']: item['caption'] for item in data}
    return captions

class SelectedImagesDataset(DatasetFolder):
    def __init__(self, root, num_images, transform=None):
        super().__init__(root, loader=pil_loader, extensions=('jpg', 'jpeg', 'png', 'bmp', 'gif'), transform=transform)
        self.num_images = num_images
        self.samples, self.captions = self._select_samples_with_captions()

    def _select_samples_with_captions(self):
        selected_samples = []
        captions = {}
        self.classes = sorted(self.classes)
        print (self.classes)
        for class_index, class_name in enumerate(self.classes):
            class_samples = [s for s in self.samples if s[1] == class_index]
            selected_samples.extend(random.sample(class_samples, min(self.num_images, len(class_samples))))
            json_path = os.path.join(self.root, class_name, 'updated_data.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as file:
                    data = json.load(file)
                for item in data:
                    image_path = os.path.join(self.root, class_name, os.path.basename(item['image_path']))
                    image_id = int(os.path.basename(image_path).split('.')[0])
                    captions[image_id] = item['caption']
        return selected_samples, captions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image_id = int(path.split('/')[-1].split('.')[0])
        caption = self.captions.get(image_id, "")
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, caption
    
def main(args):
    for model_name in req_models:
        name = model_name
        print('Current model: ' + str(name))

        disp_name = name
        if('/' in name):
            disp_name = name.replace('/', '')

        model, preprocess = clip.load(name, download_root=CLIP_MODEL_PATH, device='cuda')
        model.eval()

        input_resolution = model.visual.input_resolution
        context_length = model.context_length
        vocab_size = model.vocab_size

        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)

        print('Processing current dataset: '+ args.dataset)

        # features_path = FEATURE_PATH.format(args.dataset) + "/" +  str(args.images_per_class)
        features_path = FEATURE_PATH.format(args.dataset, args.prompt_length)

        if not os.path.exists(features_path):
            os.makedirs(features_path)

        store_features_path = features_path + STORE_FEATURE_DIR.format(args.dataset, disp_name)
        store_targets_path = features_path + STORE_TARGET_DIR.format(args.dataset, disp_name)
        store_caption_features_path = features_path + STORE_CAPTION_FEATURE_DIR.format(args.dataset, disp_name)

        if(os.path.exists(store_features_path) and os.path.exists(store_targets_path)):
            load_stoch = True
        else:
            load_stoch = False

        caps_path = CAPS_PATH.format(args.prompt_length, args.dataset)

        sd_images = SelectedImagesDataset(caps_path, args.images_per_class, transform=preprocess)

        print ('Number of images: ' + str(len(sd_images)))

        dataloader = torch.utils.data.DataLoader(sd_images, batch_size=args.val_batch_size, num_workers=8, shuffle=False)        

        # ------------------------------------------saving features------------------------------------------
        print('start saving sus sd image features')

        if args.regenerate:
            if os.path.exists(store_features_path):
                os.remove(store_features_path)
            if os.path.exists(store_targets_path):
                os.remove(store_targets_path)
            if os.path.exists(store_caption_features_path):
                os.remove(store_caption_features_path)
            load_stoch = False
        else:
            load_stoch = os.path.exists(store_features_path) and os.path.exists(store_targets_path) and os.path.exists(store_caption_features_path)

        if not load_stoch:

            train_images_targets = []
            train_images_features_agg = []
            train_captions_features_agg = []

            # take average of features over multiple augmentations for a more robust feature set
            with torch.no_grad():
                for augment_idx in range(args.augment_epoch):
                    train_images_features = []
                    train_captions_features = []

                    print('Augment time: {:} / {:}'.format(augment_idx, args.augment_epoch))
                    for i, (images, target, captions) in enumerate(tqdm(dataloader)):
                        images = images.cuda()
                        image_features = model.encode_image(images)
                        train_images_features.append(image_features)

                        tokens = clip.tokenize([captions[k] for k in range(len(images))], truncate=True).to('cuda')
                        caption_features = model.encode_text(tokens)
                        train_captions_features.append(caption_features)

                        if augment_idx == 0:
                            target = target.cuda()
                            train_images_targets.append(target)

                    images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
                    train_images_features_agg.append(images_features_cat)
                    captions_features_cat = torch.cat(train_captions_features, dim=0).unsqueeze(0)
                    train_captions_features_agg.append(captions_features_cat)
                
            # concatenate and take mean of features from multiple augment runs
            train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(dim=0)
            train_captions_features_agg = torch.cat(train_captions_features_agg, dim=0).mean(dim=0)
            # L2 normalise image embeddings from few shot dataset -- dim NKxC
            train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
            train_captions_features_agg /= train_captions_features_agg.norm(dim=-1, keepdim=True)
            # dim CxNK
            train_images_features_agg = train_images_features_agg.permute(1, 0)
            train_captions_features_agg = train_captions_features_agg.permute(1, 0)

            # convert all image labels to one hot labels -- dim NKxN
            train_images_targets = F.one_hot(torch.cat(train_images_targets, dim=0)).half()

            assert train_images_features_agg.shape[0]==feat_dims[name], 'train_images_features_agg is not of shape CxNK'

            print('Storing features to: '+store_features_path+' and '+store_targets_path)
            # dim CxNK
            torch.save(train_images_features_agg, store_features_path)
            torch.save(train_captions_features_agg, store_caption_features_path)
            # dim NKxN
            torch.save(train_images_targets, store_targets_path)

        else:
            print('Loading features from: ' + store_features_path + ' and ' + store_targets_path + ' ,image per clas: '+ {args.images_per_class})
            # dim CxNK
            train_images_features_agg = torch.load(store_features_path)
            # dim NKxN
            train_images_targets = torch.load(store_targets_path)


if __name__ == '__main__':
    args = parse_args()

    # dummy parameters for dataloader
    args.k_shot = 2
    args.val_batch_size = 64
    args.train_batch_size = 256

    main(args)