# CapS-Adapter: Caption-based MultiModal Adapter in Zero-Shot Classification

Recent advances in vision-language foundational models, such as CLIP, have demonstrated significant strides in zero-shot classification. However, the extensive parameterization of models like CLIP necessitates a resource-intensive fine-tuning process. In response, TIP-Adapter and SuS-X have introduced training-free methods aimed at bolstering the efficacy of downstream tasks. While these approaches incorporate support sets to maintain data distribution consistency between knowledge cache and test sets, they often fall short in terms of generalization on the test set, particularly when faced with test data exhibiting substantial distributional variations. In this work, we present CapS-Adapter, an innovative method that employs a caption-based support set, effectively harnessing both image and caption features to exceed existing state-of-the-art techniques in training-free scenarios. CapS-Adapter adeptly constructs support sets that closely mirror target distributions, utilizing instance-level distribution features extracted from multimodal large models. By leveraging CLIP's single and cross-modal strengths, CapS-Adapter enhances predictive accuracy through the use of multimodal support sets. Our method achieves outstanding zero-shot classification results across 19 benchmark datasets, improving accuracy by 2.19\% over the previous leading method. Our contributions are substantiated through extensive validation on multiple benchmark datasets, demonstrating superior performance and robust generalization capabilities.

<img src="https://raw.githubusercontent.com/WLuLi/CapS-Adapter/main/figure/caps_adapter.png" alt="CapS-Adapter diagram" style="width:75%;"/>

## Getting started

Our code was tested on Python 3.8.18 and PyTorch 1.13.1+cu117.

#### Setting up environments
We recommend setting up a conda virtual environment and installing all the requirements. You can follow these steps to set up environment correctly:

```bash
git clone https://github.com/WLuLi/CapS-Adapter.git
cd caps

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda activate

conda env create -f environment.yml
```

Afterward, you can activate the conda environment using
```bash
conda activate caps
```

#### Preparing datasets
We provide detailed instructions on how to prepare all the needed datasets in [`DATA.md`](https://github.com/WLuLi/CapS-Adapter/blob/main/DATA.md).

## Running the baselines

#### Zero-shot CLIP
You can run Zero-shot CLIP inference using:
```bash
python run_zs_baseline.py --dataset <dataset> --backbone <CLIP_visual_backbone>
```
The `backbone` parameter can be one of [`RN50`, `RN101`, `ViT-B/32`, `ViT-B/16`, `all`](`all` for all these 5 backbones).

#### CuPL
For ensuring reproducibility, we used *CuPL* prompt files generated by [SuS-X](https://github.com/vishaal27/SuS-X) in [`gpt3-prompts`](https://github.com/WLuLi/CapS-Adapter/blob/main/gpt3_prompts). These prompts are used for CuPL and CuPL+e inference and text classifier CuPL and CuPL+e.

You can run the CuPL and CuPL+e baselines using:
```bash
python run_cupl_baseline.py --dataset <dataset> --backbone <CLIP_visual_backbone>
```

## CapS construction
We provide scripts for CapS Construction.

#### Generation captions
For generating captions, we need to sample from the target dataset first. You can run the following script to extract `k` samples from each class in `dataset`:

```bash
python datasampler.py --k <k> --dataset <dataset_name>
```

To generate captions using [ShareCaptioner](https://huggingface.co/Lin-Chen/ShareCaptioner), we deploy it locally.  You can run the following script to automatically download ShareCaptioner to `./models/share_captioner` and generate captions for the samples from each class in `dataset`:

```bash
python generate_captions.py --dataset <dataset>
```

#### Generate CapS images
For generating images using the [Stable-Diffusion v1-4 checkpoint](https://huggingface.co/CompVis/stable-diffusion-v1-4), we need a huggingface token. Please create an account on huggingface and find your token under the [access tokens tab](https://huggingface.co/settings/tokens).

You can run the following script to generate `images_number` images for each class in `dataset`:

```bash
python generate_caps.py --dataset <dataset> --huggingface_key <huggingface_key> --images_per_class <images_number>
```

## Constructing the features

#### Test, validation features
You can create the test and validation image features using:

```bash
python encode_datasets.py --dataset <dataset>
```

This script will save the test, validation features in `./data/features/<dataset>`.

#### Text classifier weights
You can create the different text classifier weights using:

```bash
python generate_text_classifier_weights.py --dataset <dataset>
```

This script will again save all the text classifier weights in `./data/features/<dataset>`.

It should be noted that the features for the two text classifier modes, CuPL and CuPL+e, need to be generated by running the script `run_cupl_baseline.py`.

#### CapS features
You can create the CapS features (features of `images_number` images for each class in `dataset`) using:

```bash
python encode_caps.py --dataset <dataset> --images_per_class <images_number> --regenerate
```
These scripts will also save the CapS image and text features in `./data/features/<dataset>`. If the command line argument `regenerate` is added, it will clear the existing feature files in `./data/features/<dataset>` and regenerate them.

#### Few shot weights
You can create the few shot weights (1,2,4,8,16 shots) for `<dataset>` using:

```bash
python encode_few_shot.py --dataset <dataset> --regenerate
```

If the command line argument `regenerate` is added, it will clear the existing feature files in `./data/features/few_features/<dataset>` and regenerate them.

It should be noted that madapter requires caption features for inference. Therefore, `encode_few_shot.py` randomly extracts k-shots images from pictures that have already generated captions. Please ensure that `datasampler.py` and `generate_captions.py` has been run beforehand.

## M-Adapter Inference
Once you have correctly saved all the feature files, you can run M-Adapter using:

```bash
python madapter.py --dataset <dataset> --backbone <CLIP_visual_backbone> --sus_type <sus_type> --log_file_path <filepath>
```

or

```bash
python madapter.py --dataset <dataset> --backbone <CLIP_visual_backbone> --sus_type <sus_type> --k <k> --log_file_path <filepath>
```

The `sus_type` parameter is `caps` for using CapS and `fewshot` for using few shot features. The parameter `k` is only effective in the few-shot setting, representing k-shot.  The `log_file_path` parameter is used to specify the output path for results of `madapter.py`.

## Citation
If you found this work useful, please consider citing it as:

## Acknowledgements
We build on several previous well-maintained repositories like [CLIP](https://github.com/openai/CLIP/tree/main/clip), [CoOp](https://github.com/KaiyangZhou/CoOp/), [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter), [TIP-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/), [CuPL](https://github.com/sarahpratt/CuPL) and [SuS-X](https://github.com/vishaal27/SuS-X). We thank the authors for providing such amazing code, and enabling further research towards better vision-language model adaptation. We also thank the authors of the amazing [Stable-Diffusion](https://stability.ai/blog/stable-diffusion-public-release), which is pivotal component of our method.

## Bugs or questions?
If you have any questions about the code, feel free to open an issue on the GitHub repository or email us at [wangqj20@mails.tsinghua.eud.cn](mailto:wangqj20@mails.tsinghua.eud.cn).
