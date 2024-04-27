# Setting up the datasets

We require all our datasets to be under `./data/datasets` in the project root folder. The `./data/datasets` folder should look like this:

```
data/
|–– birdsnap/
|–– caltech101/
|–– ...
```

In case you need to download your datasets to an external device or have them already downloaded at another location, you can simply create symbolic links inside `./data/datasets` pointing to the correct dataset location using:

```bash
ln -s /path/to/existing/dataset ./data/datasets/dataset
```

To ensure reproducibility and consistency to prior works, we utilize the val/test splits used in [SuS-X](https://github.com/vishaal27/SuS-X/blob/main/data/DATA.md).

### UCF101
- Create a folder named `ucf101/` under `./data/datasets`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `./data/datasets/ucf101/`. This zip file contains the extracted middle video frames.
- Download `split_zhou_UCF101.json` from this [link](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing) and put it under `./data/datasets/ucf101`.

The directory structure should look like
```
ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```

### CIFAR-10
- Create a folder named `cifar10/` under `./data/datasets`.
- The `dataloader` script will automatically download the CIFAR-10 dataset to this directory using the Pytorch dataloader.

The directory structure should look like
```
cifar10/
|–– cifar-10-batches-py
|–– cifar-10-python.tar.gz
```

### CIFAR-100
- Create a folder named `cifar100/` under `./data/datasets`.
- The `dataloader` script will automatically download the CIFAR-100 dataset to this directory using the Pytorch dataloader.

The directory structure should look like
```
cifar100/
|–– cifar-100-python
|–– cifar-100-python.tar.gz
```

### Caltech101
- Create a folder named `caltech101/` under `./data/datasets`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `./data/datasets/caltech101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `./data/datasets/caltech101`. 

The directory structure should look like
```
caltech101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### Caltech256
- Create a folder named `caltech256/` under `./data/datasets`.
- Download `256_ObjectCategories.tar` from https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar and extract the file under `./data/datasets/caltech256`.
- Download `split_Caltech256.json` from this [link](https://drive.google.com/file/d/12tQM2F3IQfH2J3YU_ul7Tt4bUR-4xVeA/view?usp=sharing) and put it under `./data/datasets/caltech256`. 

The directory structure should look like
```
caltech256/
|–– 256_ObjectCategories/
|–– split_Caltech256.json
```

### ImageNet
- Create a folder named `imagenet/` under `./data/datasets`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `./data/datasets/imagenet`. 

The directory structure should look like
```
imagenet/
|–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|–– val/
```

### SUN397
- Create a folder named  `sun397/` under `./data/datasets`.
- Download the images http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
- Download the partitions https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
- Extract these files under `./data/datasets/sun397/`.
- Download `split_zhou_SUN397.json` from this [link](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing) and put it under `./data/datasets/sun397`.

The directory structure should look like
```
sun397/
|–– SUN397/
|–– split_zhou_SUN397.json
|–– ... # a bunch of .txt files
```

### FGVCAircraft
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `./data/datasets` and rename the folder to `fgvcaircraft/`.

The directory structure should look like
```
fgvcaircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### Birdsnap
- Download the data from http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz.
- Extract `birdsnap.tgz` and ensure that it contains the `get-birdsnap.py` script.
- Run the `get-birdsnap.py` script resulting in the creation of a folder named `download`.
- Move `download/` to `./data/datasets` and rename the folder to `birdsnap/`.
- Download `split_Birdsnap.json` from this [link](https://drive.google.com/file/d/1bYfMD-DPm51c0VsaXUKa0DFwKPytYRRK/view?usp=sharing) and put it under `./data/datasets/birdsnap`.

- If the `get-birdsnap.py` script doesn't work well:
- Download all `*.tar` files from this [link](https://huggingface.co/datasets/HuggingFaceM4/Birdsnap/tree/main/images) and put it under `./data/datasets/birdsnap/images`.
- Run `./scripts/unzip_dataset_tar.sh` to unzip all `*.tar` files.
- You can delete all tars by running `./scripts/dataset_tar_delete.sh` if you want.
- Download `split_Birdsnap.json` from this [link](https://drive.google.com/file/d/1bYfMD-DPm51c0VsaXUKa0DFwKPytYRRK/view?usp=sharing) and put it under `./data/datasets/birdsnap`.

The directory structure should look like
```
birdsnap/
|–– images/
|–– split_Birdsnap.json
|–– ... # a bunch of .txt files
```

### StanfordCars
- Create a folder named `stanfordcars/` under `./data/datasets`.
- Download the train images http://ai.stanford.edu/~jkrause/car196/cars_train.tgz.
- Download the test images http://ai.stanford.edu/~jkrause/car196/cars_test.tgz.
- Download the train labels https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz.
- Download the test labels http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat.
- Download `split_zhou_StanfordCars.json` from this [link](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing) and put it under `./data/datasets/stanfordcars`.

The directory structure should look like
```
stanfordcars/
|–– cars_test/
|–– cars_test_annos_withlabels.mat
|–– cars_train/
|–– devkit/
|–– split_zhou_StanfordCars.json
```

### CUB
- Download the data from https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
- Extract CUB_200_2011 and keep only the `CUB_200_2011/` subfolder inside the extracted folder
- Move `CUB_200_2011/` to `./data/datasets` and rename the folder to `cub`.
- Download `split_CUB.json` from this [link](https://drive.google.com/file/d/1HFkRyDA0P8acHRHwiTaJi9BYvEiNxQRv/view?usp=sharing) and put it under `./data/datasets/cub`.

The directory structure should look like
```
cub/
|–– images/
|–– parts/
|–– attributes/
|–– split_CUB.json
|–– ... # a bunch of .txt files
```

### Flowers102
- Create a folder named `flowers102/` under `./data/datasets`.
- Download the images and labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz and https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat respectively.
- Download `cat_to_name.json` from [here](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing) and put it under `./data/datasets/flowers102`. 
- Download `split_zhou_OxfordFlowers.json` from [here](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing) and put it under `./data/datasets/flowers102`.

The directory structure should look like
```
flowers102/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### Food101
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `./data/datasets`, resulting in a folder named `./data/datasets/food-101/`.
- Rename `./data/datasets/food-101` to `./data/datasets/food101`.
- Download `split_zhou_Food101.json` from [here](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing) and put it under `./data/datasets/food101`.

The directory structure should look like
```
food101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

### OxfordPets
- Create a folder named `oxfordpets/` under `./data/datasets`.
- Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
- Download `split_zhou_OxfordPets.json` from this [link](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing) and put it under `./data/datasets/oxfordpets`. 

The directory structure should look like
```
oxfordpets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### DTD
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `./data/datasets`. This should lead to `./data/datasets/dtd/`.
- Download `split_zhou_DescribableTextures.json` from this [link](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing) and put it under `./data/datasets/dtd`.

The directory structure should look like
```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT
- Create a folder named `eurosat/` under `./data/datasets`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `./data/datasets/eurosat/`.
- Download `split_zhou_EuroSAT.json` from [here](https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing) and put it under `./data/datasets/eurosat`.

The directory structure should look like
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### ImageNet-Sketch
- Download the dataset from https://github.com/HaohanWang/ImageNet-Sketch.
- Extract the dataset to `./data/datasets/imagenet-sketch`.
- Download `classnames.txt` to `./data/datasets/imagenet-sketch/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).

The directory structure should look like
```
imagenet-sketch/
|–– images/ # contains 1,000 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-R
- Create a folder named `imagenet-r/` under `./data/datasets`.
- Download the dataset from https://github.com/hendrycks/imagenet-r and extract it to `./data/datasets/imagenet-r/`.
- Copy `./data/datasets/imagenet-sketch/classnames.txt` to `./data/datasets/imagenet-r/`.

The directory structure should look like
```
imagenet-r/
|–– imagenet-r/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
```

### Country211
- Create a folder named `country211` under `./data/datasets`.
- Download the dataset following the instructions in https://github.com/openai/CLIP/blob/main/data/country211.md and extract it under `./data/datasets/country211`.
- Download the metadata text file from [here](https://drive.google.com/file/d/15N7Yy5vPZWZHiuyZuux6187sAz0pr2w3/view?usp=sharing) and put it under `./data/datasets/country211`.
- Download the metadata python script from [here](https://drive.google.com/file/d/1f53d30Vg_pw0i0J2DBNBtOAUwdpp9og-/view?usp=sharing) and put it under `./data/datasets/country211`.

The directory structure should look like
```
country211/
|–– test
|–– train
|–– valid
|–– country-iso-mapping.txt
|–– country_iso_mapping.py

```

## Acknowledgements

This README has been adapted from the amazing READMEs prepared by:
* [CoOP](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)
* [TIP-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/blob/fcb06059457a3b74e44ddb0d5c96d2ea7e4c5957/DATASET.md)
* [SuS-X](https://github.com/vishaal27/SuS-X/blob/main/data/DATA.md)
