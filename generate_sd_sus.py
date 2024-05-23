# Script to generate SuS-SD support sets

import os

from diffusers import StableDiffusionPipeline
import numpy as np
import torch
import shutil
import argparse
import json
import random
from tqdm import tqdm
from utils.prompts_helper import return_photo_prompts
from torch.multiprocessing import Process, set_start_method
from utils import utils
from dataloader import KShotDataLoader

GPUS = '0,1,2,3,4,5,6,7'
HUGGINGFACE_KEY = ''
SD_CACHE_DIR = './models/stable-diffusion-v1-4'

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpus', help='Comma separated list of GPU IDs to use', type=str, default=GPUS)
	parser.add_argument('--start_index', help='Starting class index for downloading images', type=int, default=0)
	parser.add_argument('--end_index', help='Ending class index for downloading images', type=int, default=1000)
	parser.add_argument('--guidance_scale', help='Stable Diffusion guidance scale', type=float, default=9.5)
	parser.add_argument('--num_inference_steps', help='Number of denoising steps', type=int, default=85)
	parser.add_argument('--num_images', help='Number of images per class to download', type=int, default=100)
	parser.add_argument('--batch_size', help='Batch size for each Stable Diffusion inference', type=int, default=5)
	parser.add_argument('--dataset', help='Dataset to download', type=str, default='cifar10')
	parser.add_argument('--prompt_shorthand', help='Name of sub-directory for storing the dataset based on prompt', type=str, default='cupl')
	parser.add_argument('--huggingface_key', help='Huggingface key', type=str, default=HUGGINGFACE_KEY)
	parser.add_argument('--cache_dir', help='Directory to store pre-trained stable diffusion model weights', type=str, default=SD_CACHE_DIR)
	args = parser.parse_args()
	assert args.end_index>args.start_index, 'end_index is less than or equal to start_index'
	return args

def is_black_image(img, threshold=5):
    """Check if an image is almost entirely black."""
    img_array = np.array(img)
    return np.mean(img_array) < threshold

def generate_prompt_array(args, class_name, batch_size, gpt3_prompts):
	if(args.prompt_shorthand == 'photo'):
		pr_ = return_photo_prompts(args.dataset).format(class_name)
		prompt = [pr_] * batch_size
	elif(args.prompt_shorthand == 'cupl'):
		if(args.batch_size>len(gpt3_prompts[class_name])):
			prompt = gpt3_prompts[class_name] + random.sample(gpt3_prompts[class_name], args.batch_size-len(gpt3_prompts[class_name]))
		else:
			prompt = random.sample(gpt3_prompts[class_name], args.batch_size)

	assert len(prompt) == args.batch_size, 'Prompt array is larger than batch size'
	return prompt

def get_save_folder_and_class_names(args):

	if(args.dataset=='imagenet'):
		imagenet_dir = './data/datasets/imagenet/train'
		imagenet_synsets = os.listdir(imagenet_dir)
		imagenet_classes = [utils.synset_to_class_map[im] for im in imagenet_synsets]
		return imagenet_synsets, imagenet_classes
	elif(args.dataset=='cifar10'):
		string_classnames = utils.cifar10_classes()
		return string_classnames, string_classnames
	elif(args.dataset=='cifar100'):
		string_classnames = utils.cifar100_classes()
		string_classnames = [s.replace('_', ' ') for s in string_classnames]
		return string_classnames, string_classnames
	else:
		_, _, _, _, _, _, _, _, string_classnames = KShotDataLoader(args, None).load_dataset()
		string_classnames = [s.replace('_', ' ') for s in string_classnames]
		return string_classnames, string_classnames

def setup_stable_diffusion_pipeline(args, device):
    """Setup the Stable Diffusion pipeline."""
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=args.huggingface_key, safety_checker=None, cache_dir=args.cache_dir)
    pipe = pipe.to(device)
    return pipe

def worker(gpu_id, args, class_indices):
    """Worker function to generate images on a specific GPU, processing a specified list of class indices."""
    device = f'cuda:{gpu_id}'
    pipe = setup_stable_diffusion_pipeline(args, device)

    print('Started worker with GPU: {}'.format(gpu_id))

    save_folders, class_names = get_save_folder_and_class_names(args)
    gpt3_prompts = None
    if(args.prompt_shorthand == 'cupl'):
        gpt3_prompts = json.load(open('./gpt3_prompts/CuPL_prompts_{}.json'.format(args.dataset)))

    stable_diff_gen_dir = './data/caps/sus-sd/{}/{}'.format(args.dataset, args.prompt_shorthand)

    num_images = args.num_images
    batch_size = args.batch_size

    for ind in class_indices:
        save_folder = save_folders[ind]
        class_name = class_names[ind]
        print(f'Started class {ind}: {class_name}: {save_folder}')
        print('Started class {}: {}: {}'.format(ind, class_name, save_folder))

        class_metadata_path = os.path.join(stable_diff_gen_dir, save_folder, 'metadata.json')

        if not os.path.exists(os.path.join(stable_diff_gen_dir, save_folder)):
            os.makedirs(os.path.join(stable_diff_gen_dir, save_folder))
        elif not os.path.exists(class_metadata_path):
            shutil.rmtree(os.path.join(stable_diff_gen_dir, save_folder))
            os.makedirs(os.path.join(stable_diff_gen_dir, save_folder))

        if os.path.exists(class_metadata_path):
            with open(class_metadata_path, 'r') as json_file:
                class_metadata = json.load(json_file)
        else:
            class_metadata = []

        files_curr = os.listdir(os.path.join(stable_diff_gen_dir, save_folder))
        if len(files_curr) >= num_images:
            print('Class {}: {}: {} already contains {} images or more'.format(ind, class_name, save_folder, str(num_images)))
            continue
        else:
            num_images = num_images - len(class_metadata)

        for batch_ind in range(num_images // batch_size):
            prompt = generate_prompt_array(args, class_name, batch_size, gpt3_prompts)
            generator = torch.Generator(device=device).manual_seed(batch_ind)
            output = pipe(prompt, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, generator=generator)
            images = output.images

            for image_ind, img in enumerate(images):
                while is_black_image(img):
                    prompt = generate_prompt_array(args, class_name, 1, gpt3_prompts)
                    generator = torch.Generator(device=pipe.device).manual_seed(torch.randint(0, 10000, (1,)).item())
                    output = pipe(prompt, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, generator=generator)
                    img = output.images[0]

                image_filename = '{}_{}.JPEG'.format(batch_ind, image_ind)
                img_path = os.path.join(stable_diff_gen_dir, save_folder, image_filename)
                img.save(img_path, 'JPEG')
                class_metadata.append({
                    'id': '{}_{}'.format(batch_ind, image_ind),
                    'image_path': img_path,
                    'prompt': prompt[image_ind]
                })

        with open(class_metadata_path, 'w') as json_file:
            json.dump(class_metadata, json_file, indent=4)

        print('Finished class {}: {}: {}'.format(ind, class_name, save_folder))


def main(args):
    """Main function to distribute image generation across specified GPUs with non-sequential class allocation."""
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',') if gpu.strip()]

    save_folders, class_names = get_save_folder_and_class_names(args)
    total_classes = len(class_names)
    
    gpu_to_classes = {gpu_id: [] for gpu_id in gpu_ids}
    for class_index, gpu_id in enumerate(gpu_ids * (total_classes // len(gpu_ids) + 1)):
        if class_index < total_classes:
            gpu_to_classes[gpu_id].append(class_index)

    processes = []
    for gpu_id, class_indices in gpu_to_classes.items():
        print ('GPU ID:', gpu_id, 'Class Indices:', class_indices)
        p = Process(target=worker, args=(gpu_id, args, class_indices))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':

	args = parse_args()

	# dummy parameters for dataloader
	args.k_shot = 2
	args.val_batch_size = 64 
	args.train_batch_size = 256

	try:
		set_start_method('spawn')
	except RuntimeError:
		pass

	main(args)