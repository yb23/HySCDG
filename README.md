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
