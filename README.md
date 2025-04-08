<div align="center">
<h1>The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation [CVPR 2025]</h1>

<a href="https://arxiv.org/pdf/2503.15683" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-8A2BE2" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2503.15683"><img src="https://img.shields.io/badge/arXiv-2503.15683-b31b1b" alt="arXiv"></a>
<a href="https://yb23.github.io/projects/cywd/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/Yanis236/HySCDG'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_SD_+_CN-blue'></a>
<a href='https://huggingface.co/Yanis236/FSC-Pretrained'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pretrained_DualUnet-blue'></a>
<a href='https://huggingface.co/datasets/Yanis236/fsc-180k'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset_FSC_180k-b31b1b'></a>

**[Univ Gustave Eiffel, ENSG, IGN, LASTIG](https://www.umr-lastig.fr/)**;


[Yanis Benidir](https://yb23.github.io/), [Nicolas Gonthier](https://ngonthier.github.io/), [Clément Mallet](https://www.umr-lastig.fr/clement-mallet/)


<br>
**If you ❤️ or simply use this project, don't forget to give the repository a ⭐, it means a lot to us !**
<br>
</div>

```bibtex
@inproceedings{benidir2025cywd,
  title={The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation},
  author={Benidir, Yanis and Gonthier, Nicolas and Mallet, Clément},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Overview

This repository is the official implementation to run the experiment of "The Change You Want to Detect" paper. It can be used to run the transfert learning experiments and evaluation on 5 different change detection dataset but also to generate new pair of images with change

## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub). 

```bash
git clone git@github.com:yb23/HySCDG.git
cd HySCDG
pip install -r requirements.txt
```

## Transfer Learning Experiments

The following commands can fine-tune the pretrained change detection model for different transfer learning settings. 

**Sequential mode: Pretraining + Fine-tuning**
```
python train.py --sequential --batch=8 -mc --classes=20 --in_channels=3 --logdir=/my-folder/logs/ -p="MyWandbProject" --log_images_every=1 --model="unet" --pretrain_name="fsc" --pretrain_path=datasets/fsc-180k/ --fsc_versions 11 12 --mix_fsc_versions --epochs=10 --target_name="hiucd" --target_path=datasets/hiucd_mini_512/ --epochs_finetune=50 --new_n_classes=10
```


**Sequential mode: Only Fine-tuning**
```
python train.py --sequential --no_pretrain --batch=8 -mc --classes=20 --in_channels=3 --logdir=/my-folder/logs/ -p="MyWandbProject" --log_images_every=1 --model="unet" --pretrain_name="fsc" --pretrain_path=datasets/fsc-180k/ --fsc_versions 11 12 --mix_fsc_versions --epochs=0 --target_name="hiucd" --target_path=datasets/hiucd_mini_512/ --epochs_finetune=50 --new_n_classes=10 --run_id="WANDB_ID_OF_PRETRAINING_RUN"
```


**Mixed mode: Pretraining x Fine-tuning**
```
python train.py --mixed --mix_ratio=0.5 --batch=8 -mc --classes=10 --in_channels=3 --logdir=/my-folder/logs/ -p="MyWandbProject" --log_images_every=1 --model="unet" --pretrain_name="fsc" --pretrain_path=datasets/fsc-180k/ --fsc_versions 11 12 --mix_fsc_versions --epochs=10 --target_name="hiucd" --target_path=datasets/hiucd_mini_512/
```


**Low Data Regime**
```
python train.py --sequential --target_max_proportion=0.10 --batch=8 -mc --classes=20 --in_channels=3 --logdir=/my-folder/logs/ -p="MyWandbProject" --log_images_every=1 --model="unet" --pretrain_name="fsc" --pretrain_path=datasets/fsc-180k/ --fsc_versions 11 12 --mix_fsc_versions --epochs=10 --target_name="hiucd" --target_path=datasets/hiucd_mini_512/ --epochs_finetune=200 --new_n_classes=10
```


**Other tips**

+ `--resume` and `--run_id` : If you want to continue an interrupted run, you have to provide the run_id of the run as well as the `--resume` argument (`--run_id="WANDB_RUN_ID" --resume`). If you just want to initialize the model weights from a specific checkpoint, you must only provide the `--run_id`. It can be either a WANDB_RUN_ID or directly a path to a .ckpt file.


## Hybrid Semantic Change Dataset Generation

The following command can be used to generate new samples from the hybrid change detection dataset. 

```
python generate.py --model_path="path_to_inpainting_pipeline/trained_pipeline" --controlnet_path="path_to_controlnet/trained_controlnet" --batch=1 --images_path="path_to_flair/flair_aerial_train" --save_dir="../data/CHG" --prompts_path="../data/FLAIR_Prompts.csv" --dfobjects_path="../data/instancesFootprints.pkl" --num_version=15
```

**If you need to run the code on CPU**, simply add the --cpu flag.

**Needed data**
The provided code is adapted to FLAIR data.
+ FLAIR dataset is publicly available here: https://ignf.github.io/FLAIR/index.html
+ The files containing the prompts (FLAIR_Prompts.csv) and the instances footprints (instancesFootprints.pkl) are available on Zenodo : https://zenodo.org/records/15129648

If you want to use another dataset as a basis for generation, you will have to adapt the "FLAIR_Dataset" class to the structure of your data and use your own prompts. Don't hesitate to reach out to us if you need further information.

## Useful links 

- The FSC-180k dataset can be found at : https://huggingface.co/datasets/Yanis236/fsc-180k
- The weights of the DualUnet pretrained on FSC-180k are available at : https://huggingface.co/Yanis236/FSC-Pretrained
- The weights of the generation model (Stable Diffusion + ControlNet) are available at : https://huggingface.co/Yanis236/HySCDG
