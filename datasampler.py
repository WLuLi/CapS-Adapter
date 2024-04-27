# Script for sampling all 19 datasets

import os
import hashlib
import numpy as np
import torchvision
from collections import defaultdict
import argparse
import json
from utils import utils
import random
from tqdm import tqdm

DATASET_PATH = './data/datasets/{}'
SAMPLE_PATH = './data/caps/samples/{}'

class KShotDataSampler():
    def __init__(self, args):
        self.dataset_path = DATASET_PATH.format(args.dataset)
        self.args = args

    def extract_id_from_path(self, image_path):
        """
        Extracts an id from the image path
        """
        hash_object = hashlib.sha256(image_path.encode())
        return int(hash_object.hexdigest()[:8], 16)

    def create_sample_json(self, sampled_data):
        dataset_dir = SAMPLE_PATH.format(self.args.dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        for class_name, images in sampled_data.items():
            formatted_class_name = class_name.replace(" ", "_").replace("/", "")
            class_dir = os.path.join(dataset_dir, formatted_class_name)
            os.makedirs(class_dir, exist_ok=True)

            json_path = os.path.join(class_dir, "sampled_data.json")
            
            json_data = [{"id": self.extract_id_from_path(image_path), "image_path": image_path} for image_path in images]

            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)

            print(f'Sampled image paths for class {class_name} saved to {json_path}')

    def parse_image_paths(self, dataset_path, splits_paths):
        train_split = splits_paths['train']

        train_class_to_images_map = {}
        train_classnames = []

        for ind in train_split:
            train_image_path = ind[0]
            train_label = ind[1]
            train_classname = ind[2]

            if(train_label in train_class_to_images_map):
                train_class_to_images_map[train_label].append(os.path.join(dataset_path, train_image_path))
            else:
                train_class_to_images_map[train_label] = []
                train_class_to_images_map[train_label].append(os.path.join(dataset_path, train_image_path))
            # train_classnames.append((train_label, train_classname))
            train_classnames.append(train_classname)

        unique_classnames = list(set(train_classnames))
        unique_classnames.sort()

        assert len(unique_classnames) == utils.get_num_classes(self.args.dataset), 'Total num classes is not correct'

        return train_class_to_images_map, unique_classnames

    def sample_dataset(self):
        if(self.args.dataset == 'imagenet'):
            return self.imagenet_sample()
        elif(self.args.dataset == 'imagenet-r'):
            return self.imagenet_r_sample()
        elif(self.args.dataset == 'imagenet-sketch'):
            return self.imagenet_sketch_sample()
        elif(self.args.dataset == 'stanfordcars'):
            return self.custom_sample()
        elif(self.args.dataset == 'ucf101'):
            return self.custom_sample()
        elif(self.args.dataset == 'caltech101'):
            return self.custom_sample()
        elif(self.args.dataset == 'caltech256'):
            return self.custom_sample()
        elif(self.args.dataset == 'cub'):
            return self.custom_sample()
        elif(self.args.dataset == 'country211'):
            return self.country_211_sample()
        elif(self.args.dataset == 'flowers102'):
            return self.custom_sample()
        elif(self.args.dataset == 'sun397'):
            return self.custom_sample()
        elif(self.args.dataset == 'dtd'):
            return self.custom_sample()
        elif(self.args.dataset == 'eurosat'):
            return self.custom_sample()
        elif(self.args.dataset == 'fgvcaircraft'):
            return self.fgvcaircraft_sample()
        elif(self.args.dataset == 'oxfordpets'):
            return self.custom_sample()
        elif(self.args.dataset == 'food101'):
            return self.custom_sample()
        elif(self.args.dataset == 'birdsnap'):
            return self.custom_sample()
        elif(self.args.dataset == 'cifar10'):
            return self.cifar10_sample()
        elif(self.args.dataset == 'cifar100'):
            return self.cifar100_sample()
        else:
            raise ValueError('Dataset not supported')
        
    def country_211_sample(self):

        traindir = os.path.join(self.dataset_path, 'train')
        train_images = torchvision.datasets.ImageFolder(traindir)
        num_classes = len(list(np.unique(train_images.targets)))
        string_classnames = utils.country211_classes()

        assert len(list(np.unique(train_images.targets))) == len(string_classnames), 'train image targets length is not country211 classes'

        split_by_label_dict = defaultdict(list)
        print('Load Country211 data finished.')

        for i in range(len(train_images.imgs)):
            split_by_label_dict[train_images.targets[i]].append(train_images.imgs[i][0])
        
        sampled_data = {}

        for label, items in split_by_label_dict.items():
            sampled_paths = random.sample(items, min(self.args.k, len(items)))
            class_name = string_classnames[label]
            sampled_data[class_name] = sampled_paths
        print('Sample Country211 data finished.')

        return sampled_data, num_classes, string_classnames
    
    def imagenet_sample(self):

        traindir = os.path.join(self.dataset_path, 'train')
        train_images = torchvision.datasets.ImageFolder(traindir)
        num_classes = len(list(np.unique(train_images.targets)))

        string_classnames = utils.imagenet_classes()

        assert len(list(np.unique(train_images.targets))) == len(string_classnames), 'train image targets length is not imagenet classes'

        split_by_label_dict = defaultdict(list)
        print('Load Imagenet data finished.')

        for i in range(len(train_images.imgs)):
            split_by_label_dict[train_images.targets[i]].append(train_images.imgs[i][0])
        
        sampled_data = {}
        for label, items in split_by_label_dict.items():
            sampled_paths = random.sample(items, min(self.args.k, len(items)))
            class_name = string_classnames[label]
            sampled_data[class_name] = sampled_paths

        print('Sample Imagenet data finished.')

        return sampled_data, num_classes, string_classnames

    def imagenet_r_sample(self):

        valdir = os.path.join(self.dataset_path, 'imagenet-r')
        val_images = torchvision.datasets.ImageFolder(valdir)
        num_classes = len(list(np.unique(val_images.targets)))

        string_classnames = utils.imagenet_r_classes()
        assert len(list(np.unique(val_images.targets))) == len(string_classnames), 'val image targets length is not imagenet-r classes'

        split_by_label_dict = defaultdict(list)
        print('Load Imagenet-R data finished.')

        for i in range(len(val_images.imgs)):
            split_by_label_dict[val_images.targets[i]].append(val_images.imgs[i][0])

        sampled_data = {}
        for label, items in split_by_label_dict.items():
            sampled_paths = random.sample(items, min(self.args.k, len(items)))
            class_name = string_classnames[label]
            sampled_data[class_name] = sampled_paths

        print('Sample Imagenet-R data finished.')

        return sampled_data, num_classes, string_classnames

    def imagenet_sketch_sample(self):

        valdir = os.path.join(self.dataset_path, 'images')
        val_images = torchvision.datasets.ImageFolder(valdir)
        
        num_classes = 1000
        string_classnames = utils.imagenet_classes()

        assert len(list(np.unique(val_images.targets))) == len(string_classnames), 'val image targets length is not imagenet classes'

        split_by_label_dict = defaultdict(list)
        print('Load Imagenet-Sketch data finished.')

        for i in range(len(val_images.imgs)):
            split_by_label_dict[val_images.targets[i]].append(val_images.imgs[i][0])

        sampled_data = {}

        for label, items in split_by_label_dict.items():
            sampled_paths = random.sample(items, min(self.args.k, len(items)))
            class_name = string_classnames[label]
            sampled_data[class_name] = sampled_paths
        
        print('Sample Imagenet-Sketch data finished.')

        return sampled_data, num_classes, string_classnames

    def cifar10_sample(self):

        trainset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=True)
        string_classnames = utils.cifar10_classes()
        num_classes = len(string_classnames)

        sample_dir = os.path.join(self.dataset_path,"sampled_data")
        print (sample_dir)
        os.makedirs(sample_dir, exist_ok=True)
        
        split_by_label_dict = defaultdict(list)
        for i in tqdm(range(len(trainset)), ascii=True):
            img, label = trainset[i]
            img_path = os.path.join(sample_dir, f"class_{label}_{i}.png")
            img.save(img_path)
            split_by_label_dict[label].append(img_path)

        print('Load CIFAR-10 data finished.')

        sampled_data = {}
        for label, items in split_by_label_dict.items():
            sampled_paths = random.sample(items, min(self.args.k, len(items)))
            class_name = string_classnames[label]
            if class_name == 'plane':
                class_name = 'airplane'
            sampled_data[class_name] = sampled_paths

        print('Sample CIFAR-10 data finished.')

        return sampled_data, num_classes, string_classnames

    def cifar100_sample(self):
            
        trainset = torchvision.datasets.CIFAR100(root=self.dataset_path, train=True, download=True)
        string_classnames = utils.cifar100_classes()
        num_classes = len(string_classnames)

        sample_dir = os.path.join(self.dataset_path,"sampled_data")
        print (sample_dir)
        os.makedirs(sample_dir, exist_ok=True)
        
        split_by_label_dict = defaultdict(list)
        for i in tqdm(range(len(trainset)), ascii=True):
            img, label = trainset[i]
            img_path = os.path.join(sample_dir, f"class_{label}_{i}.png")
            img.save(img_path)
            split_by_label_dict[label].append(img_path)

        print('Load CIFAR-100 data finished.')

        sampled_data = {}
        for label, items in split_by_label_dict.items():
            sampled_paths = random.sample(items, min(self.args.k, len(items)))
            class_name = string_classnames[label]
            sampled_data[class_name] = sampled_paths

        print('Sample CIFAR-100 data finished.')

        return sampled_data, num_classes, string_classnames

    def fgvcaircraft_sample(self):

        images_dir = os.path.join(self.dataset_path, 'images')
        train_split_image_names_file = os.path.join(self.dataset_path, 'images_variant_train.txt')
        classnames_file = os.path.join(self.dataset_path, 'variants.txt')

        label_to_classname_mapping = {}
        classname_to_label_mapping = {}

        with open(classnames_file, 'r') as f:
            string_classnames = [f.strip() for f in f.readlines()]
            for i in range(len(string_classnames)):
                label_to_classname_mapping[i] = string_classnames[i]
                classname_to_label_mapping[string_classnames[i]] = i
        
        sampled_data = {}
        class_to_samples_map = {}
        sampled_image_paths = []
        sampled_classnames = []
        sampled_labels = []

        with open(train_split_image_names_file, 'r') as f:
            paths_and_classes = f.readlines()
            paths_and_classes = [p.strip().split() for p in paths_and_classes]

            for p in paths_and_classes:
                sampled_image_paths.append(os.path.join(images_dir, p[0] + '.jpg'))
                curr_classname = ' '.join(p[1:])
                sampled_image_paths.append(curr_classname)
                sampled_labels.append(classname_to_label_mapping[curr_classname])

                if(curr_classname in class_to_samples_map):
                    class_to_samples_map[curr_classname].append(os.path.join(images_dir, p[0]+'.jpg'))
                else:
                    class_to_samples_map[curr_classname] = []
                    class_to_samples_map[curr_classname].append(os.path.join(images_dir, p[0]+'.jpg'))

        for classname, image_paths in class_to_samples_map.items():
            sampled_data[classname] = random.sample(image_paths, min(self.args.k, len(image_paths)))

        num_classes = len(string_classnames)
        
        print('Sample FGVCAircraft data finished.')

        return sampled_data, num_classes, string_classnames

    def custom_sample(self):

        if(self.args.dataset == 'stanfordcars'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_StanfordCars.json')
            root_data_dir = self.dataset_path
        elif(self.args.dataset == 'ucf101'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_UCF101.json')
            root_data_dir = os.path.join(self.dataset_path, 'UCF-101-midframes')
        elif(self.args.dataset == 'caltech101'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_Caltech101.json')
            root_data_dir = os.path.join(self.dataset_path, '101_ObjectCategories')
        elif(self.args.dataset == 'caltech256'):
            json_path = os.path.join(self.dataset_path, 'split_Caltech256.json')
            root_data_dir = os.path.join(self.dataset_path, '256_ObjectCategories')
        elif(self.args.dataset == 'cub'):
            json_path = os.path.join(self.dataset_path, 'split_CUB.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')     
        elif(self.args.dataset == 'birdsnap'):
            json_path = os.path.join(self.dataset_path, 'split_Birdsnap.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')
        elif(self.args.dataset == 'flowers102'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_OxfordFlowers.json')
            root_data_dir = os.path.join(self.dataset_path, 'jpg')
        elif(self.args.dataset == 'sun397'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_SUN397.json')
            root_data_dir = os.path.join(self.dataset_path, 'SUN397')
        elif(self.args.dataset == 'dtd'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_DescribableTextures.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')
        elif(self.args.dataset == 'eurosat'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_EuroSAT.json')
            root_data_dir = os.path.join(self.dataset_path, '2750')
        elif(self.args.dataset == 'oxfordpets'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_OxfordPets.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')
        elif(self.args.dataset == 'food101'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_Food101.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')            
        else:
            raise ValueError("Dataset not supported")

        splits_paths = json.load(open(json_path))

        train_class_to_images_map, string_classnames = self.parse_image_paths(root_data_dir, splits_paths)

        if(self.args.dataset=='caltech256'):
            string_classnames = [s.split('.')[1].replace('-101', '') for s in string_classnames]

        sampled_data = {}

        for class_id, image_paths in train_class_to_images_map.items():
            sampled_paths = random.sample(image_paths, min(self.args.k, len(image_paths)))
            class_name = string_classnames[class_id]
            sampled_data[class_name] = sampled_paths
            # print (class_name, sampled_data[class_name])

        print('Sample ' + str(self.args.dataset) + ' data finished.')

        num_classes = len(string_classnames)

        return sampled_data, num_classes, string_classnames

if __name__ == '__main__':

    # Test datasampleers for each dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='imagenet')
    args = parser.parse_args()

    k_dl = KShotDataSampler(args)
    sampled_data, num_classes, string_classnames = k_dl.sample_dataset()
    assert len(sampled_data) == utils.get_num_classes(args.dataset)
    k_dl.create_sample_json(sampled_data)
