import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.multiprocessing import Process, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

GPUS = '0,1,2,3,4,5,6,7'
DATASET_PATH = './data/caps/sample/{}'
OUTPUT_BASE = './data/caps/captioned_sample/{}'
CAPTION_MODEL_PATH = './models/share_captioner'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-name", type=str, default=CAPTION_MODEL_PATH)
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--gpus", type=str, default=GPUS, help="Comma separated list of GPU IDs to use")
    return parser.parse_args()

def generate_captions_for_class(model, class_path, seg_emb1, seg_emb2, args, device):
    with open(class_path, 'r') as file:
        imgs = json.load(file)

    captions = []
    chunk_size = len(imgs) // args.batch_size
    if len(imgs) % args.batch_size != 0:
        chunk_size += 1

    for i in range(chunk_size):
        print(f'{i}/{chunk_size}')
        subs = []
        for j in range(args.batch_size):
            if i * args.batch_size + j < len(imgs):
                img_data = imgs[i * args.batch_size + j]
                img_path = img_data["image_path"]
                image = Image.open(img_path).convert("RGB")
                subs.append(model.vis_processor(image).unsqueeze(0))

        if not subs:
            break

        subs = torch.cat(subs, dim=0).to(device)
        tmp_bs = subs.shape[0]
        tmp_seg_emb1 = seg_emb1.repeat(tmp_bs, 1, 1).to(device)
        tmp_seg_emb2 = seg_emb2.repeat(tmp_bs, 1, 1).to(device)

        with torch.cuda.amp.autocast(), torch.no_grad():
            subs = model.encode_img(subs)
            input_emb = torch.cat([tmp_seg_emb1, subs, tmp_seg_emb2], dim=1)
            out_embeds = model.internlm_model.generate(
                inputs_embeds=input_emb,
                max_length=500,
                num_beams=3,
                min_length=1,
                do_sample=True,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1.,
                eos_token_id=model.tokenizer.eos_token_id,
                num_return_sequences=1
            )

        for j, out in enumerate(out_embeds):
            out[out == -1] = 2
            response = model.decode_text([out])
            img_data = imgs[i * args.batch_size + j]
            captions.append({"id": img_data["id"], "image_path": img_data["image_path"], "caption": response})

    return captions

def worker(gpu, args, class_indices):
    torch.cuda.set_device(gpu)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True).to(gpu).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
    model.tokenizer = tokenizer
    model.half()

    seg1 = '<|User|>:'
    seg2 = f'Generate a concise and accurate description for the following image. Please ensure to include key elements and any details.\n "!{model.eoh}\n<|Bot|>:'
    seg_emb1 = model.encode_text(seg1, add_special_tokens=True)
    seg_emb2 = model.encode_text(seg2, add_special_tokens=False)

    dataset_path = DATASET_PATH.format(args.dataset)
    output_base = OUTPUT_BASE.format(args.dataset)
    classes = os.listdir(dataset_path)

    if gpu == 0:
        progress_bar = tqdm(range(class_indices), desc="Generating captions")
    else:
        progress_bar = range(class_indices)

    for index in progress_bar:
        if index < len(classes):
            class_name = classes[index]
            class_path = os.path.join(dataset_path, class_name, "sampled_data.json")
            output_path = os.path.join(output_base, class_name, "captions.json")

            if os.path.exists(output_path):
                print(f"Captions already generated for {class_name}, skipping.")
                continue

            if os.path.isfile(class_path):
                captions = generate_captions_for_class(model, class_path, seg_emb1, seg_emb2, args, gpu)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(captions, f, indent=4)
                print(f'Captions saved for {class_name}')

if __name__ == '__main__':
    args = parse_args()
    gpu_ids = [int(id) for id in args.gpus.split(',') if id.strip()]
    num_gpus = len(gpu_ids)
    print (f'Using {num_gpus} GPUs')

    classes = os.listdir(DATASET_PATH.format(args.dataset))
    num_classes = len(classes)
    
    class_indices = {gpu: [] for gpu in gpu_ids}
    for i, class_name in enumerate(classes):
        assigned_gpu = gpu_ids[i % num_gpus]
        class_indices[assigned_gpu].append(i)

    processes = []
    for gpu_id in gpu_ids:
        p = Process(target=worker, args=(gpu_id, args, class_indices[gpu_id]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()