# HySCDG


## Usage

**Sequential mode : Pretraining + Fine-tuning**
```
python train.py --sequential --batch=8 -mc --classes=20 --in_channels=3 --logdir=/my-folder/logs/ -p="MyWandbProject" --log_images_every=1 --model="unet" --pretrain_name="fsc" --pretrain_path=datasets/fsc-180k/ --fsc_versions 11 12 --mix_fsc_versions --epochs=10 --target_name="hiucd" --target_path=datasets/hiucd_mini_512/ --epochs_finetune=50 --new_n_classes=10
```


**Sequential mode : Only Fine-tuning**
```
python train.py --sequential --no_pretrain --batch=8 -mc --classes=20 --in_channels=3 --logdir=/my-folder/logs/ -p="MyWandbProject" --log_images_every=1 --model="unet" --pretrain_name="fsc" --pretrain_path=datasets/fsc-180k/ --fsc_versions 11 12 --mix_fsc_versions --epochs=0 --target_name="hiucd" --target_path=datasets/hiucd_mini_512/ --epochs_finetune=50 --new_n_classes=10 --run_id="WANDB_ID_OF_PRETRAINING_RUN"
```


**Mixed mode : Pretraining x Fine-tuning**
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

```
python generate.py --model_path="path_to_inpainting_pipeline/trained_pipeline" --controlnet_path="path_to_controlnet/trained_controlnet" --batch=1 --images_path="path_to_flair/flair_aerial_train" --save_dir="../data/CHG" --prompts_path="../data/FLAIR_Prompts.csv" --dfobjects_path="../data/instancesFootprints.pkl" --num_version=15
```

**If you need to run the code on CPU**, simply add the --cpu flag.

**Needed data**\
The provided code is adapted to FLAIR data.\
+ FLAIR dataset is publicly available here : https://ignf.github.io/FLAIR/index.html
+ The files containing the prompts (FLAIR_Prompts.csv) and the instances footprints (instancesFootprints.pkl) are available on Zenodo : https://zenodo.org/records/15129648

If you want to use an other dataset as a basis for generation, you will have to adapt the "FLAIR_Dataset" class to the structure of your data and use your own prompts. Don't hesitate to contact us if you need further information.