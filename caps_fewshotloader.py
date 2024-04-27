import os
import torch
import json
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image

TRAIN_BATCH_SIZE = 256

def get_transform(dataset_name):
    if dataset_name == 'country211':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    elif dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    elif dataset_name == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    elif dataset_name == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    elif dataset_name == 'fgvcaircraft':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    else:   
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    return train_transform

class ImageDatasetFromPaths(Dataset):
    def __init__(self, split_entity, transform):
        self.image_paths, self.captions, self.labels = split_entity.image_paths, split_entity.captions, split_entity.labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        label = self.labels[idx]

        try:
            image = read_image(img_path)
        except RuntimeError as e:
            # HACK: if the image is corrupted or not readable, then sample a random image
            image_rand = None
            while(image_rand is None):
                rand_ind = random.randint(0, self.__len__())
                try:
                    image_rand = read_image(self.image_paths[rand_ind])
                except RuntimeError as e1:
                    image_rand = None
                    continue
            image = image_rand
            caption = self.captions[rand_ind]
            label = self.labels[rand_ind]

        image = transforms.ToPILImage()(image)
        image = image.convert("RGB")

        if(self.transform):
            image = self.transform(image)
        return image, caption, label

class DataEntity():
    def __init__(self, image_paths, captions, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.captions = captions

class SelectedImagesLoader():
    def __init__(self, root, k_shot, dataset):
        self.dataset = dataset
        self.transform = get_transform(dataset)
        self.root = root
        self.k_shot = k_shot
        self.classes = os.listdir(root)

    def _select_samples_with_captions(self):
        img_paths = []
        targets = []
        captions = []

        self.classes = sorted(self.classes)
        if self.dataset == 'country211':
            class_text = 'A photo shoot in {}. '
        elif self.dataset == 'eurosat':
            class_text = 'A satellite image of {}. '
        else:
            class_text = 'A photo of {}. '
        for class_index, class_name in enumerate(self.classes):
            # if self.dataset == 'cifar10' and class_name == 'plane':
            #     class_name = 'airplane'
            class_text = class_text.format(class_name)
            json_path = os.path.join(self.root, class_name, 'captions.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as file:
                    data = json.load(file)
                selected_items = random.sample(data, min(len(data), self.k_shot))
                for item in selected_items:
                    img_paths.append(item['image_path'])
                    targets.append(class_index)
                    captions.append(class_text + item['caption'])
        
        train_dataset = ImageDatasetFromPaths(DataEntity(img_paths, captions, targets), transform=self.transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=8, shuffle=False)
        return train_loader