# Script to encode the SuS-SD images using CLIP's image encoders

import os
import numpy as np
import torch
import clip
from caps_fewshotloader import SelectedImagesLoader
import torch.nn.functional as F
import random
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

CLIP_MODEL_PATH = './models'
FEATURE_PATH = './features/few_features/{}'
CAPTION_FEATURE_DIR = '/{}_caption_features_m{}_k{}.pt'
IMAGE_FEATURE_DIR = '/{}_image_features_m{}_k{}.pt'
TARGET_DIR = '/{}_target_m{}_k{}.pt'
CAPTION_DATA_PATH = "./data/caps/captioned_sample/{}"

# feature dimensions for each model
feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

# models to encode images with
req_models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
req_shots = [1, 2, 4, 8, 16]

random.seed(1)
torch.manual_seed(1)

def parse_args():
    parser = argparse.ArgumentParser()
    # number of augmentations to apply for averaging visual features
    parser.add_argument('--augment_epoch', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate features')
    args = parser.parse_args()
    return args

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def main(args):
    for shot in req_shots:
        for model_name in req_models:
            args.k_shot = shot
            args.backbone = model_name

            name = model_name
            print('Current model: {} and k-shot: {}'.format(model_name, shot))

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

            features_path = FEATURE_PATH.format(args.dataset)

            if not os.path.exists(features_path):
                os.makedirs(features_path)

            images_features_path = features_path + IMAGE_FEATURE_DIR.format(args.dataset, disp_name, args.k_shot)
            caption_features_path = features_path + CAPTION_FEATURE_DIR.format(args.dataset, disp_name, args.k_shot)
            targets_path = features_path + TARGET_DIR.format(args.dataset, disp_name, args.k_shot)

            if(os.path.exists(images_features_path) and os.path.exists(caption_features_path) and os.path.exists(targets_path)):
                load_stoch = True
            else:
                load_stoch = False

            captioned_data_path = CAPTION_DATA_PATH.format(args.dataset)

            dataloader = SelectedImagesLoader(captioned_data_path, args.k_shot, args.dataset)._select_samples_with_captions()        

            # ------------------------------------------saving features------------------------------------------
            print('start saving few-shot features')

            if args.regenerate:
                if os.path.exists(images_features_path):
                    os.remove(images_features_path)
                if os.path.exists(caption_features_path):
                    os.remove(caption_features_path)
                if os.path.exists(targets_path):
                    os.remove(targets_path)
                load_stoch = False
            else:
                load_stoch = os.path.exists(images_features_path) and os.path.exists(caption_features_path) and os.path.exists(targets_path)

            if not load_stoch:

                images_features_agg = []
                captions_features_agg = []
                images_targets = []

                # take average of features over multiple augmentations for a more robust feature set
                with torch.no_grad():
                    for augment_idx in range(args.augment_epoch):
                        images_features = []
                        captions_features = []

                        print('Augment time: {:} / {:}'.format(augment_idx, args.augment_epoch))
                        for i, (images, captions, targets) in enumerate(tqdm(dataloader)):
                            images = images.cuda()
                            image_features = model.encode_image(images)
                            images_features.append(image_features)

                            tokens = clip.tokenize([captions[k] for k in range(len(images))], truncate=True).to('cuda')
                            caption_features = model.encode_text(tokens)
                            captions_features.append(caption_features)

                            if augment_idx == 0:
                                targets = targets.cuda()
                                images_targets.append(targets)

                        images_features_cat = torch.cat(images_features, dim=0).unsqueeze(0)
                        images_features_agg.append(images_features_cat)
                        captions_features_cat = torch.cat(captions_features, dim=0).unsqueeze(0)
                        captions_features_agg.append(captions_features_cat)
                    
                # concatenate and take mean of features from multiple augment runs
                images_features_agg = torch.cat(images_features_agg, dim=0).mean(dim=0)
                captions_features_agg = torch.cat(captions_features_agg, dim=0).mean(dim=0)
                # L2 normalise image embeddings from few shot dataset -- dim NKxC
                images_features_agg /= images_features_agg.norm(dim=-1, keepdim=True)
                captions_features_agg /= captions_features_agg.norm(dim=-1, keepdim=True)
                # dim CxNK
                images_features_agg = images_features_agg.permute(1, 0)
                captions_features_agg = captions_features_agg.permute(1, 0)

                # convert all image labels to one hot labels -- dim NKxN
                images_targets = F.one_hot(torch.cat(images_targets, dim=0)).half()

                assert images_features_agg.shape[0]==feat_dims[name], 'images_features_agg is not of shape CxNK'

                print('Storing features to: '+images_features_path+' and '+targets_path+' and '+caption_features_path)
                # dim CxNK
                torch.save(images_features_agg, images_features_path)
                torch.save(captions_features_agg, caption_features_path)
                # dim NKxN
                torch.save(images_targets, targets_path)

            else:
                print('Loading features from: ' + images_features_path + ' and ' + targets_path + ' and ' + caption_features_path)


if __name__ == '__main__':
    args = parse_args()

    # dummy parameters for dataloader
    args.batch_size = 64

    main(args)