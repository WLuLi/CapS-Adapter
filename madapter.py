# Main implementation of M-Adapter
# We use some parts of the SuS-X codebase: https://github.com/vishaal27/SuS-X

import os
import gc
import torch
import clip
import torch.nn as nn
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import argparse
import logging
from utils import utils

CLIP_MODEL_PATH = './models'
FEATURE_PATH = './data/features/'

random.seed(1)
torch.manual_seed(1)

def compute_image_text_distributions(temp, features, vanilla_zeroshot_weights, batch_size):
    batches = []
    for i in range((features.shape[0]+batch_size-1)//batch_size):
        start = i * batch_size
        end = min((i+1) * batch_size, features.shape[0])
        curr_batch = features[start:end]
        batch_result = curr_batch @ vanilla_zeroshot_weights
        batch_result = nn.Softmax(dim=-1)(curr_batch/temp)
        batches.append(batch_result)

    return torch.cat(batches, dim=0)

def get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution):
    bs = 10
    kl_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

    for i in tqdm(range(test_image_class_distribution.shape[0]//bs)):
        curr_batch = test_image_class_distribution[i*bs : (i+1)*bs]
        repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)    
        q = train_image_class_distribution
        q_repeated = torch.cat([q]*bs)
        kl = repeated_batch * (repeated_batch.log() - q_repeated.log())
        kl = kl.sum(dim=-1)
        kl = kl.view(bs, -1)
        kl_divs_sim[ i*bs : (i+1)*bs , : ] = kl  

    return kl_divs_sim

def get_kl_div_sims(args, test_features, val_features, train_features, clip_weights):
    batch_size = 1000
    train_image_class_distribution = compute_image_text_distributions(args.temperature, train_features.T, clip_weights, batch_size)
    test_image_class_distribution = compute_image_text_distributions(args.temperature, test_features, clip_weights, batch_size)
    val_image_class_distribution = compute_image_text_distributions(args.temperature, val_features, clip_weights, batch_size)

    train_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, train_image_class_distribution)
    test_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution)
    val_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, val_image_class_distribution)

    return train_kl_divs_sim, test_kl_divs_sim, val_kl_divs_sim

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def scale_(x, target):
    
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()
    
    return y

def hparam_search(val_features, val_labels, test_features, test_labels, train_images_features_agg, train_captions_features_agg, train_images_targets, zeroshot_weights, val_kl_divs_sim, test_kl_divs_sim):

    search_scale = [50, 50, 30, 10]
    search_step = [200, 20, 50, 11]

    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]
    beta_list = [i * (search_scale[0] - 1) / search_step[0] + 1 for i in range(search_step[0])]
    gamma_list = [i * (search_scale[2] - 0.1) / search_step[2] + 0.1 for i in range(search_step[2])]
    delta_list = [i / search_scale[3] for i in range(search_step[3])]

    best_tipx_acc = 0 

    best_gamma_tipx, best_alpha_tipx, best_beta_tipx, best_delta_tipx = 0, 0, 0, 0

    hsearch_batch_size = 8192
    num_batches = (val_features.size(0) + hsearch_batch_size - 1) // hsearch_batch_size

    for alpha in tqdm(alpha_list, desc="Alpha Progress"):
        for beta in beta_list:
            for delta in delta_list:
                n = 0.
                total_clip_logits = []
                total_kl_logits = []
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

                    neg_affs = scale_((val_kl_divs_sim[start_idx:end_idx]).cuda(), new_knowledge)
                    affinities = -neg_affs
                    kl_logits = affinities.half() @ train_images_targets
                    total_kl_logits.append(kl_logits)

                    n += val_features_batch.size(0)

                total_cache_logits = torch.cat(total_cache_logits, dim=0)
                total_clip_logits = torch.cat(total_clip_logits, dim=0)
                total_kl_logits = torch.cat(total_kl_logits, dim=0)

                for gamma in gamma_list:  
                    tipx_top1, tipx_top5 = 0., 0.

                    tipx_logits = total_clip_logits + total_kl_logits * gamma + total_cache_logits * alpha
                    tipx_acc1, tipx_acc5 = accuracy(tipx_logits, val_labels, topk=(1, 5))
                    tipx_top1 += tipx_acc1
                    tipx_top5 += tipx_acc5
                    tipx_top1 = (tipx_top1 / n) * 100
                    tipx_top5 = (tipx_top5 / n) * 100

                    if tipx_top1 > best_tipx_acc:
                        best_tipx_acc = tipx_top1
                        best_alpha_tipx = alpha
                        best_gamma_tipx = gamma
                        best_beta_tipx = beta
                        best_delta_tipx = delta

    del val_features
    del val_kl_divs_sim
    del total_clip_logits
    del total_kl_logits
    del total_cache_logits
    gc.collect()

    n = 0.
    total_clip_logits = []
    total_kl_logits = []
    total_cache_logits = []

    test_batch_size = 8192
    num_batches = (test_features.size(0) + test_batch_size - 1) // test_batch_size
    for b in range(num_batches):
        start_idx = b * test_batch_size
        end_idx = min((b + 1) * test_batch_size, test_features.size(0))
        test_features_batch = test_features[start_idx:end_idx]

        new_knowledge = (1 - best_delta_tipx) * test_features_batch @ train_images_features_agg + best_delta_tipx * test_features_batch @ train_captions_features_agg
        
        cache_logits = ((-1) * (best_beta_tipx - best_beta_tipx * new_knowledge)).exp() @ train_images_targets
        clip_logits = 100. * test_features_batch @ zeroshot_weights

        total_cache_logits.append(cache_logits)
        total_clip_logits.append(clip_logits)

        neg_affs = scale_((test_kl_divs_sim[start_idx:end_idx]).cuda(), new_knowledge)
        affinities = -neg_affs
        kl_logits = affinities.half() @ train_images_targets
        total_kl_logits.append(kl_logits)

        n += test_features_batch.size(0)
    
    total_cache_logits = torch.cat(total_cache_logits, dim=0)
    total_clip_logits = torch.cat(total_clip_logits, dim=0)
    total_kl_logits = torch.cat(total_kl_logits, dim=0)

    tipx_top1, tipx_top5 = 0., 0.

    tipx_logits = total_clip_logits + total_kl_logits * best_gamma_tipx + total_cache_logits * best_alpha_tipx
    tipx_acc1, tipx_acc5 = accuracy(tipx_logits, test_labels, topk=(1, 5))
    tipx_top1 += tipx_acc1
    tipx_top5 += tipx_acc5
    tipx_top1 = (tipx_top1 / n) * 100
    tipx_top5 = (tipx_top5 / n) * 100

    # logging.info('Best params searched. alpha: {}, beta: {}, gamma: {}, best delta: {}, TIP-X-C Accuracy: {}'.format(best_alpha_tipx, best_beta_tipx, best_gamma_tipx, best_delta_tipx, tipx_top1))

    return tipx_top1, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, best_delta_tipx

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='all')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--sus_type', type=str, default='caps')
    parser.add_argument('--text_prompt_type', type=str, default='combined')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--log_file_path', type=str, default=None)
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

        train_kl_divs_sims, test_kl_divs_sims, val_kl_divs_sims = get_kl_div_sims(args, test_features, val_features, support_features, text_classifier_weights)

        tipx_acc, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, best_delta_tipx = hparam_search(val_features, val_labels, test_features, test_labels, support_features, support_caption_features, support_labels, text_classifier_weights, val_kl_divs_sims, test_kl_divs_sims)

        print('--------------------------------------------')
        print('Best for Dataset: {}, Model: {}, SuS Type: {}, k: {}, alpha: {}, beta: {}, gamma: {}, delta: {}, CapS-Adapter Accuracy: {}'.format(args.dataset, args.backbone, args.sus_type, args.k, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, best_delta_tipx, tipx_acc))
        logging.info('Best for Dataset: {}, Model: {}, SuS Type: {}, k: {}, alpha: {}, beta: {}, gamma: {}, delta: {}, CapS-Adapter Accuracy: {}'.format(args.dataset, args.backbone, args.sus_type, args.k, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, best_delta_tipx, tipx_acc))
        print('--------------------------------------------')
    print()
    print('----------------------------------------------------------------------------')

    logging.info('----------------------------------------------------------------------------')
