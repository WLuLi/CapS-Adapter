# Main implementation of few-shot TIP-Adapter
# We use some parts of the TIP-Adapter codebase: https://github.com/gaopengcuhk/Tip-Adapter

import os
import random
import argparse
import yaml
import torch

import clip
from utils_tip import *

import logging
import time

random.seed(1)
torch.manual_seed(1)

FEW_FEATURES_PATH = "./data/features/few_features/{}"
FEATURES_PATH = "./data/features/{}"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()
    return args

def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = 100. * val_features @ clip_weights + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    if cfg['dataset'] == 'imagenet':
        best_beta, best_alpha = cfg['init_beta'], cfg['init_alpha']
    else:
        best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Tip-Adapter
    tip_start_time = time.time()

    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = 100. * test_features @ clip_weights + cache_logits * best_alpha
    tip_acc = cls_acc(tip_logits, test_labels)

    tip_end_time = time.time()
    tip_inference_time = tip_end_time - tip_start_time

    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(tip_acc))
    print(f"Tip-Adapter Inference Time: {tip_inference_time} seconds")
    return tip_acc, tip_inference_time

def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print("\nRunning configs.")
    print(cfg, "\n")

    dataset = cfg['dataset']

    logging.basicConfig(filename='tip_adapter_' + dataset + '.log', filemode='a', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    backbones = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'] if cfg['backbone'] == 'all' else cfg['backbone']
    shots_list = [1, 2, 4, 8, 16] if cfg['shots'] == 'all' else [int(cfg['shots'])]

    for backbone in backbones:
        for shots in shots_list:
            shots = shots
            disp_name = backbone.replace('/', '')

            # Prepare dataset
            print("Preparing dataset.")

            few_features_path = FEW_FEATURES_PATH.format(dataset)
            features_path = FEATURES_PATH.format(dataset)

            cache_values = torch.load(os.path.join(few_features_path, '{}_target_m{}_k{}.pt'.format(dataset, disp_name, shots)))
            cache_keys = torch.load(os.path.join(few_features_path, '{}_image_features_m{}_k{}.pt'.format(dataset, disp_name, shots)))

            # Pre-load val features
            print("\nLoading visual features and labels from val set.")
            val_features = torch.load(os.path.join(features_path, "{}_f_val_m{}.pt".format(dataset, disp_name)))
            val_labels = torch.load(os.path.join(features_path, "{}_t_val_m{}.pt".format(dataset, disp_name)))

            # Pre-load test features
            print("\nLoading visual features and labels from test set.")
            test_features = torch.load(os.path.join(features_path, "{}_f_test_m{}.pt".format(dataset, disp_name)))
            test_labels = torch.load(os.path.join(features_path, "{}_t_test_m{}.pt".format(dataset, disp_name)))

            # Load text classifier weights
            text_classifier_weights = torch.load(os.path.join(features_path, '{}_zeroshot_text_weights_m{}_pt{}.pt'.format(dataset, disp_name, cfg['text_prompt_type'])))

            # ------------------------------------------ Tip-Adapter ------------------------------------------
            tip_acc, tip_inference_time = run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, text_classifier_weights)

            logging.info('Best for Dataset: {}, Model: {}, k: {}, TIP-Adapter Accuracy: {}, Total Inference Time: {} seconds, Test Size: {}'.format(dataset, backbone, shots, tip_acc, tip_inference_time, len(test_features)))

if __name__ == '__main__':
    main()
