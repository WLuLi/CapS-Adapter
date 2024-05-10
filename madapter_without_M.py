# Main implementation of M-Adapter
# We use some parts of the SuS-X codebase: https://github.com/vishaal27/SuS-X

import os
import gc
import torch
import clip
import torch.nn as nn
import random
from tqdm import tqdm
import argparse
import logging
from utils import utils

CLIP_MODEL_PATH = './models'
FEATURE_PATH = './data/features/'

random.seed(1)
torch.manual_seed(1)

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def scale_(x, target):
    
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()
    
    return y

def hparam_search(val_features, val_labels, test_features, test_labels, train_images_features_agg, train_captions_features_agg, train_images_targets, zeroshot_weights):

    search_scale = [50, 50, 10]
    search_step = [200, 20, 11]

    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]
    beta_list = [i * (search_scale[0] - 1) / search_step[0] + 1 for i in range(search_step[0])]
    delta_list = [i / search_scale[2] for i in range(search_step[2])]

    best_tipx_acc = 0 

    best_alpha_tipx, best_beta_tipx, best_delta_tipx = 0, 0, 0

    hsearch_batch_size = 8192
    num_batches = (val_features.size(0) + hsearch_batch_size - 1) // hsearch_batch_size

    for alpha in tqdm(alpha_list, desc="Alpha Progress"):
        for beta in beta_list:
            for delta in delta_list:
                n = val_features.size(0)
                total_clip_logits = []
                total_cache_logits = []

                for b in range(num_batches):
                    start_idx = b * hsearch_batch_size
                    end_idx = min((b + 1) * hsearch_batch_size, val_features.size(0))
                    val_features_batch = val_features[start_idx:end_idx]

                    new_knowledge = (1 - delta) * val_features_batch @ train_images_features_agg + delta * val_features_batch @ train_captions_features_agg
                    
                    cache_logits = ((-1) * (beta - beta * new_knowledge)).exp() @ (train_images_targets)
                    clip_logits = 100. * val_features_batch @ zeroshot_weights

                    total_cache_logits.append(cache_logits)
                    total_clip_logits.append(clip_logits)

                total_cache_logits = torch.cat(total_cache_logits, dim=0)
                total_clip_logits = torch.cat(total_clip_logits, dim=0)
                  
                tipx_top1, tipx_top5 = 0., 0.

                tipx_logits = total_clip_logits + total_cache_logits * alpha
                tipx_acc1, tipx_acc5 = accuracy(tipx_logits, val_labels, topk=(1, 5))
                tipx_top1 += tipx_acc1
                tipx_top5 += tipx_acc5
                tipx_top1 = (tipx_top1 / n) * 100
                tipx_top5 = (tipx_top5 / n) * 100

                if tipx_top1 > best_tipx_acc:
                    best_tipx_acc = tipx_top1
                    best_alpha_tipx = alpha
                    best_beta_tipx = beta
                    best_delta_tipx = delta

    del val_features
    del total_clip_logits
    del total_cache_logits
    gc.collect()
                           
    n = test_features.size(0)

    test_batch_size = 8192
    num_batches = (n + test_batch_size - 1) // test_batch_size

    total_clip_logits = []
    total_cache_logits = []

    for b in range(num_batches):
        start_idx = b * test_batch_size
        end_idx = min((b + 1) * test_batch_size, test_features.size(0))
        test_features_batch = test_features[start_idx:end_idx]

        new_knowledge = (1 - best_delta_tipx) * test_features_batch @ train_images_features_agg + best_delta_tipx * test_features_batch @ train_captions_features_agg
        
        cache_logits = ((-1) * (best_beta_tipx - best_beta_tipx * new_knowledge)).exp() @ train_images_targets
        clip_logits = 100. * test_features_batch @ zeroshot_weights

        total_cache_logits.append(cache_logits)
        total_clip_logits.append(clip_logits)

    total_cache_logits = torch.cat(total_cache_logits, dim=0)
    total_clip_logits = torch.cat(total_clip_logits, dim=0)

    tipx_top1, tipx_top5 = 0., 0.

    tipx_logits = clip_logits + cache_logits * best_alpha_tipx
    tipx_acc1, tipx_acc5 = accuracy(tipx_logits, test_labels, topk=(1, 5))
    tipx_top1 += tipx_acc1
    tipx_top5 += tipx_acc5
    tipx_top1 = (tipx_top1 / n) * 100
    tipx_top5 = (tipx_top5 / n) * 100

    return tipx_top1, best_alpha_tipx, best_beta_tipx, best_delta_tipx

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='RN50')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--sus_type', type=str, default='caps')
    parser.add_argument('--text_prompt_type', type=str, default='combined')
    parser.add_argument('--log_file_path', type=str, default=None)
    parser.add_argument('--k', type=int, default=1)
    args = parser.parse_args()

    # dummy parameters for dataloader
    args.k_shot = 2
    args.val_batch_size = 64 
    args.train_batch_size = 256

    logging.basicConfig(filename=args.log_file_path, filemode='a', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

    if(args.backbone=='all'):
        req_models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    else:
        req_models = [args.backbone]
    features_path = FEATURE_PATH + args.dataset
    few_features_path = FEATURE_PATH + 'few_features/' + args.dataset

    for model_ in req_models:

        print('Current model: {}'.format(model_))

        args.backbone = model_

        clip_model, preprocess = clip.load(args.backbone, download_root=CLIP_MODEL_PATH)
        clip_model.cuda()
        clip_model.eval()

        input_resolution = clip_model.visual.input_resolution
        context_length = clip_model.context_length
        vocab_size = clip_model.vocab_size

        dataset = args.dataset
        model = args.backbone

        disp_name = model
        if('/' in model):
            disp_name = model.replace('/', '')

        feat_dim = utils.get_model_feat_dims(model)
        num_classes = utils.get_num_classes(dataset)

        val_features_path = features_path+"/{}_f_val_m{}.pt".format(dataset, disp_name)
        val_targets_path = features_path+"/{}_t_val_m{}.pt".format(dataset, disp_name)

        test_features_path = features_path+"/{}_f_test_m{}.pt".format(dataset, disp_name)
        test_targets_path = features_path+"/{}_t_test_m{}.pt".format(dataset, disp_name)

        if args.sus_type == 'caps':
            support_features_path = os.path.join(features_path, 'caps_{}_f_m{}.pt'.format(dataset, disp_name))
            support_labels_path = os.path.join(features_path, 'caps_{}_t_m{}.pt'.format(dataset, disp_name))

            support_caption_features_path = os.path.join(features_path, 'caps_{}_c_m{}.pt'.format(dataset, disp_name))
        elif args.sus_type == 'fewshot':
            support_features_path = os.path.join(few_features_path, '{}_image_features_m{}_k{}.pt'.format(dataset, disp_name, args.k))
            support_labels_path = os.path.join(few_features_path, '{}_target_m{}_k{}.pt'.format(dataset, disp_name, args.k))

            support_caption_features_path = os.path.join(few_features_path, '{}_caption_features_m{}_k{}.pt'.format(dataset, disp_name, args.k))

        text_classifier_weights_path = os.path.join(features_path, "{}_zeroshot_text_weights_m{}_pt{}.pt".format(dataset, disp_name, args.text_prompt_type))

        # dim nxC
        val_features = torch.load(val_features_path)
        # dim n
        val_labels = torch.load(val_targets_path)

        # dim nxC 
        test_features = torch.load(test_features_path)
        # dim n
        test_labels = torch.load(test_targets_path)

        # dim nxC
        support_features = torch.load(support_features_path)
        # dim n
        support_labels = torch.load(support_labels_path)

        #dim dxC
        support_caption_features = torch.load(support_caption_features_path)

        # print shape
        print ("image feature shape:", support_features.shape)
        print ("caption feature shape:", support_caption_features.shape)
        print ("label feature shape:", support_labels.shape)

        text_classifier_weights = torch.load(text_classifier_weights_path)

        tipx_acc, best_alpha_tipx, best_beta_tipx, best_delta_tipx = hparam_search(val_features, val_labels, test_features, test_labels, support_features, support_caption_features, support_labels, text_classifier_weights)

        print('--------------------------------------------')
        print('Best for Dataset: {}, Model: {}, SuS Type: {}, alpha: {}, beta: {}, delta: {}, TIP-X-C Accuracy: {}'.format(args.dataset, args.backbone, args.sus_type, best_alpha_tipx, best_beta_tipx, best_delta_tipx, tipx_acc))
        logging.info('Best for Dataset: {}, Model: {}, SuS Type: {}, Prompting strategy: {}, alpha: {}, beta: {}, delta: {}, TIP-X-C Accuracy: {}'.format(args.dataset, args.backbone, args.sus_type, best_alpha_tipx, best_beta_tipx, best_delta_tipx, tipx_acc))
        print('--------------------------------------------')
    print()
    print('----------------------------------------------------------------------------')

    logging.info('----------------------------------------------------------------------------')
