import warnings
warnings.filterwarnings('ignore')

from src import models
from src import dataset

import argparse
import pytorch_lightning as pyl
from pytorch_lightning.loggers import WandbLogger
import wandb
import glob
from pytorch_lightning.callbacks import LearningRateMonitor
import datetime
import torch
import numpy as np
import os




def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", default="TransferLearning")
    parser.add_argument("-b", "--batch", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--reducedLR", default=1, type=float)
    parser.add_argument("-l", "--local", action="store_true")
    parser.add_argument("-mc", "--multiclass", action="store_true")
    parser.add_argument("--classes", default=2, type=int)
    parser.add_argument("--in_channels", default=5, type=int)
    parser.add_argument("--n_validation", default=12, type=int, help="Number of images to log during validation step")
    parser.add_argument("--val_every", default=0, type=int)
    parser.add_argument("--log_images_every", default=2, type=int, help="Log images every x epochs")
    parser.add_argument("--restr_train_set", action="store_true", help="Restrict train set to 200 items for debugging")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--n_workers", default=8, type=int)

    parser.add_argument("--target_name", default="second")

    parser.add_argument("--mixed", action="store_true", help="Run a mixed training")
    parser.add_argument("--mix_ratio", default=0.5, type=float, help="Proportion of target elements in the pretraining dataset (only for mixed training)")
    parser.add_argument("--sequential", action="store_true", help="Run a sequential training")
    parser.add_argument("--only_pretrain", action="store_true", help="Only pretraining, no finetuning")
    parser.add_argument("--no_pretrain", action="store_true", help="No pretraining, run directly finetuning on target")
    parser.add_argument("--only_test", action="store_true", help="No pretraining, no finetuning, run directly test step on target")
    
    parser.add_argument("--augment", action="store_true", help="Apply simple augmentations on training images")
    parser.add_argument("--normalize", action="store_true")
    
    parser.add_argument("--target_max_proportion", default=1.0, type=float, help="Restrict available samples from the training set for training")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--epochs_finetune", default=1, type=int)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_id", default="", help="WandB ID of the run to resume. Only load the checkpoint if resume is False")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--new_n_classes", default=0, type=int)
    parser.add_argument("--reset_semantic_head", action="store_true")
    parser.add_argument("--reset_change_head", action="store_true")
    parser.add_argument("--use_flair_classes", action="store_true")
    parser.add_argument("--use_target_classes", action="store_true")

    parser.add_argument("--save_preds", action="store_true", default=False)
    parser.add_argument("--preds_folder", default="")
    parser.add_argument("--class_weights", action="store_true", default=False)
    parser.add_argument("--chg_weight", default=1.0, type=float)
    parser.add_argument("--seg_weight", default=1.0, type=float)

    parser.add_argument("--crop256", action="store_true", default=False)
    parser.add_argument("--noSimilarityLoss", action="store_true")
    parser.add_argument("--changeSimilarityLoss", action="store_true")
    parser.add_argument("--sgd", action="store_true", help="Use SGD rather than AdamW")

    parser.add_argument("--model", default="unet", help="Architecture name : can be either unet, mambaCD, scannet")
    
    # Pour MambaCD
    parser.add_argument("--mambaCD", action="store_true", default=False)
    parser.add_argument('--cfg', type=str, default='/home/YBenidir/Documents/Algorithmes/Flairchange/MambaCD/changedetection/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument("--opts",help="Modify config options by adding 'KEY VALUE' pairs. ",default=None,nargs='+',)
    parser.add_argument('--pretrained_weight_path', type=str, default="../pretrained/vssm_base_0229_ckpt_epoch_237.pth")

    # Pour SCanNet
    parser.add_argument("--scannet", action="store_true", default=False)