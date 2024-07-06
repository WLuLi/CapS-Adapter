import os
import json
import torch
import shutil
import logging
import numpy as np
import argparse
from utils.utils import get_num_classes
from torch.multiprocessing import Process, set_start_method
from diffusers import StableDiffusionPipeline

try:
    set_start_method('spawn')
except RuntimeError:
    pass

GPUS = '0,1,2,4,5,6,7'
HUGGINGFACE_KEY = ''
DATASET_PATH = './data/caps/captioned_sample/{}'
JSON_FILE_PATH = './data/caps/captioned_sample/{}/{}/captions.json'
OUTPUT_PATH = './data/caps/caps_sd/{}/{}/{}'
NONE_OUTPUT_PATH = './data/caps/caps_sd/{}/{}'
SD_CACHE_DIR = './models/stable-diffusion-v1-4'

logging.basicConfig(filename='generate_sd_caption.log', filemode='a', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--huggingface_key', type=str, help="Huggingface API key", default=HUGGINGFACE_KEY)
    parser.add_argument('--guidance_scale', help='Stable Diffusion guidance scale', type=float, default=9.5)
    parser.add_argument('--num_inference_steps', help='Number of denoising steps', type=int, default=85)
    parser.add_argument('--batch_size', help='Batch size for each Stable Diffusion inference', type=int, default=5)
    parser.add_argument('--cache_dir', type=str, help="Model cache directory", default=SD_CACHE_DIR)
    parser.add_argument("--gpus", type=str, default=GPUS, help="Comma separated list of GPU IDs to use")
    parser.add_argument('--images_per_class', type=int, required=True, help="Number of images to generate per class")
    parser.add_argument('--max_prompt_length', type=int, default=None, help="Maximum length of the prompt")
    return parser.parse_args()

def is_black_image(img, threshold=5):
    """Check if an image is almost entirely black."""
    img_array = np.array(img)
    return np.mean(img_array) < threshold

def is_truncated_image(img):
    """Check if an image is truncated (incomplete)."""
    try:
        img.verify()
        return False
    except IOError:
        return True

def load_json_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def check_images_count(output_dir, images_per_class):
    """
    Check if the number of images in the output directory is equal to or greater than the number of images to generate.
    """
    if not os.path.exists(output_dir):
        return False
    existing_images = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    return len(existing_images) >= images_per_class

def clear_directory(directory_path):
    """
    Clears all files and folders in the specified directory.
    """
    logging.info(f"clear {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def setup_stable_diffusion_pipeline(cache_dir, huggingface_key, gpu):
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=huggingface_key, safety_checker=None, cache_dir=cache_dir)
    pipe = pipe.to(device)
    return pipe

def generate_images(args, data, classname, dataset, pipe, output_dir, images_per_class):
    """Generate images for each class and create new entries for each image."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_len = len(data)
    selected_indices = np.arange(data_len)
    np.random.shuffle(selected_indices)

    updated_data = []
    generated_count = 0
    while generated_count < images_per_class:
        index = selected_indices[generated_count % data_len]
        original_entry = data[index]
        if original_entry is None:
            continue

        caption = ""
        classname_without_underline = classname.replace("_", " ")
        if dataset == "country211":
            caption += f"In {classname_without_underline}. "
        else:
            caption += f"A photo of {classname_without_underline}. "
        caption += original_entry["caption"]

        if args.max_prompt_length is not None:
            words = caption.split()
            if len(words) > args.max_prompt_length:
                caption = " ".join(words[:args.max_prompt_length])

        random_seed = torch.randint(0, 10000, (1,)).item()
        generator = torch.Generator(device=pipe.device).manual_seed(random_seed)

        image = pipe(caption, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, generator=generator).images[0]
        image_truncated = is_truncated_image(image)
        nsfw_count = 0
        while image_truncated:
            generator = torch.Generator(device=pipe.device).manual_seed(torch.randint(0, 10000, (1,)).item())
            image = pipe(caption, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, generator=generator).images[0]

            if is_black_image(image):
                nsfw_count += 1
                continue
            
            if nsfw_count >= 5:
                break
            image_truncated = is_truncated_image(image)

        if nsfw_count >= 5:
            continue

        image_name = f'{original_entry["id"]}_{generated_count}.png'
        image_path = os.path.join(output_dir, image_name)
        image.save(image_path)

        new_entry = original_entry.copy()
        new_entry["image_path"] = image_path
        new_entry["id"] = os.path.splitext(image_name)[0]
        new_entry["caption"] = caption
        updated_data.append(new_entry)

        generated_count += 1

    return updated_data

def worker(gpu, args, class_indices, images_per_class):
    torch.cuda.set_device(gpu)
    dataset_path = DATASET_PATH.format(args.dataset)
    num_classes = get_num_classes(args.dataset)
    class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    assert num_classes == len(class_names), f'Mismatch between number of classes and class names of {args.dataset}: {num_classes} vs {len(class_names)}'

    for index in class_indices:
        if index < len(class_names):
            classname = class_names[index]
            json_file_path = JSON_FILE_PATH.format(args.dataset, classname)
            
            if args.max_prompt_length is not None:
                output_dir = OUTPUT_PATH.format(args.max_prompt_length, args.dataset, classname)
            else :
                output_dir = NONE_OUTPUT_PATH.format(args.dataset, classname)

            if check_images_count(output_dir, images_per_class):
                print(f"Class {classname}: already has {images_per_class} or more images, skipping.")
                continue
            else:
                os.makedirs(output_dir, exist_ok=True)
                clear_directory(output_dir)
            
            if not os.path.exists(json_file_path):
                logging.info(f'not exist: {json_file_path}')
            data = load_json_data(json_file_path)
            pipe = setup_stable_diffusion_pipeline(args.cache_dir, args.huggingface_key, gpu)
            updated_data = generate_images(args, data, classname, args.dataset, pipe, output_dir, images_per_class)

            with open(os.path.join(output_dir, "updated_data.json"), 'w') as file:
                json.dump(updated_data, file, indent=4)

if __name__ == "__main__":
    args = parse_args()

    gpu_ids = [int(id) for id in args.gpus.split(',') if id.strip()]
    num_gpus = len(gpu_ids)
    classes = os.listdir(DATASET_PATH.format(args.dataset))
    num_classes = len(classes)

    class_indices = {gpu: [] for gpu in gpu_ids}
    for i, class_name in enumerate(classes):
        assigned_gpu = gpu_ids[i % num_gpus]
        class_indices[assigned_gpu].append(i)

    processes = []
    for gpu_id in gpu_ids:
        p = Process(target=worker, args=(gpu_id, args, class_indices[gpu_id], args.images_per_class))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
