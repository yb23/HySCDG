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

    parser.add_argument("--pretrain_name", default="fsc", help="Pretraining dataset name : can be either fsc, syntheworld, changen or empty string")
    parser.add_argument("--target_name", default="second")

    parser.add_argument("--pretrain_path", default="", help="Pretraining dataset root path")
    parser.add_argument("--target_path", default="", help="Target dataset root path")


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
    parser.add_argument("--chg_weight", default=1.0, type=float)
    parser.add_argument("--seg_weight", default=1.0, type=float)

    parser.add_argument("--crop256", action="store_true", default=False)
    parser.add_argument("--noSimilarityLoss", action="store_true")
    parser.add_argument("--changeSimilarityLoss", action="store_true")
    parser.add_argument("--sgd", action="store_true", help="Use SGD rather than AdamW")

    parser.add_argument("--model", default="unet", help="Architecture name : can be either unet, mambaCD, scannet")


    args = parser.parse_args()
    return args

def main():
    args  = getArgs()
    local = args.local
    start_datetime = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
    
    args.pretrain_name = args.pretrain_name if args.pretrain_name in ["fsc","syntheworld","changen"] else ""

    identifier = f"{start_datetime},{args.pretrain_name},{args.target_name},{args.model},"
    
    if args.mixed:
        identifier += f"mixed,{args.mix_ratio},"
    elif args.only_pretrain:
        run_name += f"pretrain,,"
    elif args.sequential:
        identifier += f"sequential,,"
     
    run_name = start_datetime

    identifier += f"{args.batch},"

    accelerator = "cpu" if args.cpu else "cuda"


    if args.pretrain_name != "":
        data_pretrain = dataset.loadPretrainData(args)

    print("Total Images Train : {}".format(len(data_pretrain["train"]["IMG_A"])))
    print("Total Images Val : {}".format(len(data_pretrain["val"]["IMG_A"])))

    
    ### Preparation of target Dataset
    data_target = {}
    if args.target_name != "":
        data_target = dataset.DataDict(data_path=args.target_path, dataset_name=args.target_name, file_ext="png")
        class_mapping_to = args.target_name
        n_target = len(data_target["train"]["IMG_A"]) 
        print(f"Total train images in target dataset : {n_target}")
        if args.target_max_proportion<1:
            idxs = np.random.choice(n_target, int(n_target*args.target_max_proportion))
            for key in data_target["train"]:
                data_target["train"][key] = [data_target["train"][key][ii] for ii in idxs]
            print(f"Keeping only {len(data_target['train']['IMG_A'])} images ({100*args.target_max_proportion:.0f}%)")
        ### Datamodule for validation samples (only from target dataset)
        dm_target_val = dataset.LEVIR_DataModule(dict_train=data_target["val"], dict_val=data_target["val"], dict_test=data_target["val"], batch_size=min(args.batch, args.n_validation), num_channels=args.in_channels, isBinary=(not args.multiclass), class_mapping_to=class_mapping_to, augment_first_image=args.augment, use_augmentations=args.augment, crop256=args.crop256, dataset_name=args.target_name, normalize=args.normalize, args=args)
        dm_target_val.prepare_data()
        dm_target_val.setup(stage="validate")
        val_samples = next(iter(dm_target_val.val_dataloader()))

    else:
        class_mapping_to = None
        dm_target_val = dataset.LEVIR_DataModule(dict_train=data_pretrain["val"], dict_val=data_pretrain["val"], dict_test=data_pretrain["val"], batch_size=min(args.batch, args.n_validation), num_channels=args.in_channels, isBinary=(not args.multiclass), class_mapping_to=class_mapping_to, augment_first_image=args.augment, use_augmentations=args.augment, crop256=args.crop256, dataset_name=args.pretrain_name, normalize=args.normalize, args=args)
        dm_target_val.prepare_data()
        dm_target_val.setup(stage="validate")
        val_samples = next(iter(dm_target_val.val_dataloader()))


    ### Preparation of the training dataset
    data_train = {}
    if args.mixed:
        print("Mixed Training")
        if "MSK_A" in data_pretrain["train"]:
            data_target["train"]["MSK_A"] = [""] * len(data_target["train"]["MSK"])
            data_target["val"]["MSK_A"] = [""] * len(data_target["val"]["MSK"])
        data_train["train"] = dataset.mixDicts(data_pretrain["train"], data_target["train"],args.mix_ratio)
        data_train["val"] = dataset.mixDicts(data_pretrain["val"], data_target["val"],args.mix_ratio)
        data_train["test"] = data_target["test"]
        ds_name = args.target_name
    else:
        data_train = data_pretrain
        ds_name = args.pretrain_name
    
    dm_train = dataset.LEVIR_DataModule(dict_train=data_train["train"], dict_val=data_train["val"], dict_test=data_train["test"], batch_size=args.batch, num_channels=args.in_channels, isBinary=(not args.multiclass), num_workers=args.n_workers, class_mapping_to=class_mapping_to, augment_first_image=args.augment, use_augmentations=args.augment, crop256=args.crop256, dataset_name=ds_name, normalize=args.normalize, args=args)


    ### Initialization of the model
    if args.run_id != "":
        if len((args.run_id).split("/"))>1:   # you can provide a path to a .ckpt file instead of a wandb run_id
            ckpt = args.run_id
            identifier += f"{ckpt},"
            model = models.Lightning(in_channels=args.in_channels, doSemantics=not args.only_change, ignore_index=0, num_classes=args.classes, lr=args.lr,  args=args,)
            sd = torch.load(ckpt)
            model.model.load_state_dict(sd)
        else:
            ckpts = glob.glob(f"{args.logdir}/lightning_logs/{args.run_id}/checkpoints/epoch=*.ckpt")
            ckpts.sort(key = lambda x:int(x.split("epoch=")[1].split("-")[0]))
            ckpt = ckpts[-1]
            identifier += f"{ckpt},"
            model = models.Lightning.load_from_checkpoint(ckpt, doSemantics=not args.only_change, in_channels=args.in_channels, ignore_index=0, num_classes=args.classes, lr=args.lr, args=args, strict=False)
        print(f"Loading checkpoint : {ckpt}")
        
    else:
        identifier += f","
        model = models.Lightning(in_channels=args.in_channels, doSemantics=not args.only_change, ignore_index=0, num_classes=args.classes, lr=args.lr, args=args)
        
    model.configure_losses()
    model.configure_metrics()

    if args.resume:
        run = wandb.init(id=args.run_id, resume="must", mode="offline", project=args.project, dir=args.logdir)
    else:
        run = wandb.init(name=run_name, mode="offline", project=args.project, dir=args.logdir)
    wandb_logger = WandbLogger(experiment=run, log_model=True, save_dir=args.logdir)
    
    model.identifier += f"{wandb.run.id},"
    print(f"Run Identifier : {model.identifier}")

    ### Preparation of the finetuning datamodule
    if len(data_target)>0:
        if args.no_pretrain:
            class_mapping_to = None
            print("No pretrain -> class_mapping_to=None")
        dm_target = dataset.LEVIR_DataModule(dict_train=data_target["train"], dict_val=data_target["val"], dict_test=data_target["test"], batch_size=args.batch, num_channels=args.in_channels, isBinary=(not args.multiclass), num_workers=args.n_workers, use_augmentations=args.augment, class_mapping_to=class_mapping_to, augment_first_image=False, crop256=args.crop256, dataset_name=args.target_name, normalize=args.normalize, args=args)

    ### Preparation of callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_F1_total',  # Replace with your validation metric
        mode='max',          # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
        save_top_k=1,        # Save top k checkpoints based on the monitored metric
        save_last=True,      # Save the last checkpoint at the end of training
        filename='{epoch}-{val_F1_total:.3f}'  # Checkpoint file naming pattern
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    if (args.use_flair_classes) or ((args.use_target_classes_ft) & (args.no_pretrain)):
        class_mapping_val = None
    else:
        class_mapping_val = dm_target_val.val_dataset.inv_mapping
    if args.normalize:
        norm_weights = dataset.norm_params[args.target_name] if args.target_name!="" else dataset.norm_params[args.pretrain_name]
    else:
        norm_weights = None
    imageLogger = models.ImagePredictionLoggerSemantic(val_samples, num_samples=len(val_samples["mask"]), log_every_n_epochs=args.log_images_every,  name="examples_target", class_mapping=class_mapping_val, norm_weights=norm_weights)
    callbacks = [checkpoint_callback, lr_monitor, imageLogger]

    if (test_only_change or args.only_change) and ("test" in data_target):
        dm_only_change = dataset.LEVIR_DataModule(dict_train=data_target["test"], dict_val=data_target["test"], dict_test=data_target["test"], batch_size=min(args.batch, args.n_validation), num_channels=args.in_channels, isBinary=(not args.multiclass), class_mapping_to=class_mapping_to, dataset_name=args.mix_dataset, normalize=args.normalize, args=args)
        dm_only_change.prepare_data()
        dm_only_change.setup(stage="validate")
        val_samples_only_change = next(iter(dm_only_change.val_dataloader()))
        callbacks += [models.ImagePredictionLogger(val_samples_only_change, num_samples=len(val_samples_only_change["mask"]), log_every_n_epochs=args.log_images_every, name="examples_target_change")]        
    
    if args.val_on_flair:
        dm_flair_val = dataset.LEVIR_DataModule(dict_train=data_flair["val"], dict_val=data_flair["val"], dict_test=data_flair["val"], batch_size=min(args.batch, args.n_validation), num_channels=args.in_channels, isBinary=(not args.multiclass), augment_first_image=args.augment, class_mapping_to=class_mapping_to, use_target_classes=args.use_target_classes, dataset_name="flair", normalize=args.normalize, args=args)
        dm_flair_val.prepare_data()
        dm_flair_val.setup(stage="validate")
        val_samples_flair = next(iter(dm_flair_val.val_dataloader()))
        callbacks += [models.ImagePredictionLoggerSemantic(val_samples_flair, num_samples=len(val_samples_flair["mask"]), log_every_n_epochs=args.log_images_every, name="examples_flair", norm_weights=norm_weights_flair)]        

    if args.only_test:
        if args.mix_dataset=="":
            dm_target = dm_train
        elif args.sequential:
            model.configure_finetune(only_change=(test_only_change or args.only_change or args.train_only_change), freeze_encoder=args.freeze_encoder, freeze_decoder=args.freeze_decoder, new_n_classes=args.new_n_classes, reset_semantic_head=args.reset_semantic_head, reset_change_head=args.reset_change_head)
        trainer_finetune = pyl.Trainer(accelerator=accelerator, strategy=strategy, devices=devices, logger=wandb_logger, callbacks=callbacks)
        dm_target.prepare_data()
        dm_target.setup(stage="validate")

        if args.run_id != "":
            CKPT_PATH = "/gpfswork/rech/jrj/uhz23wx/LOGS/lightning_logs/" if not local else "../data/CKPT_FlairChange/"
            try:
                ckpts = glob.glob(f"{CKPT_PATH}{args.run_id}/checkpoints/epoch=*.ckpt")
                ckpts.sort(key = lambda x:int(x.split("epoch=")[1].split("-")[0]))
                ckpt_path = ckpts[-1]
            except:
                ckpts = glob.glob(f"/lustre/fsn1/projects/rech/jrj/uhz23wx/LOGS/lightning_logs/{args.run_id}/checkpoints/epoch=*.ckpt")
                ckpts.sort(key = lambda x:int(x.split("epoch=")[1].split("-")[0]))
                ckpt_path = ckpts[-1]
            print(f"Loading checkpoint : {ckpt_path}")
        elif len((args.run_id).split("/"))>1:
            print("Using current model weights for test !")
            ckpt_path = None
            if len((args.run_id).split("/"))>1:
                ckpt = args.run_id
                sd = torch.load(ckpt)
                model.model.load_state_dict(sd)
        else:
            ckpt_path="last"
        trainer_finetune.test(model=model, ckpt_path=ckpt_path, datamodule=dm_target)
    else:

        if not args.no_pretrain:
            if len(data_train["train"]["IMG_A"])>75000:
                limit_train_batches = 75000/len(data_train["train"]["IMG_A"])
                limit_val_batches = 20000/len(data_train["val"]["IMG_A"])
                print(f"limit_train_batches : {limit_train_batches:.2f}")
                print(f"limit_val_batches : {limit_val_batches:.2f}")
                trainer = pyl.Trainer(accelerator=accelerator, strategy=strategy, devices=devices, logger=wandb_logger,callbacks=callbacks, max_epochs=args.epochs, limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches)
            else:
                trainer = pyl.Trainer(accelerator=accelerator, strategy=strategy, devices=devices, logger=wandb_logger,callbacks=callbacks, max_epochs=args.epochs)
            if args.resume:
                trainer.fit(model=model, datamodule=dm_train, ckpt_path=ckpt)
            else:
                trainer.fit(model=model, datamodule=dm_train)
            try:
                print("Test using best checkpoint !")
                trainer.test(datamodule=dm_train, ckpt_path="best")
            except:
                print("Error loading best checkpoint ! Test using last checkpoint !")
                trainer.test(datamodule=dm_train, ckpt_path="last")
            
#
        
        if args.sequential:
            model.configure_finetune(only_change=(test_only_change or args.only_change or args.train_only_change), freeze_encoder=args.freeze_encoder, freeze_decoder=args.freeze_decoder, new_n_classes=args.new_n_classes, reset_semantic_head=args.reset_semantic_head, reset_change_head=args.reset_change_head)
            print(f"Model only change : {model.train_only_change} {model.test_only_change}")
            if args.val_every>0:
                trainer_finetune = pyl.Trainer(accelerator=accelerator, strategy=strategy, devices=devices, logger=wandb_logger, callbacks=callbacks, max_epochs=args.epochs_finetune, check_val_every_n_epoch=args.val_every)
            else:
                trainer_finetune = pyl.Trainer(accelerator=accelerator, strategy=strategy, devices=devices, logger=wandb_logger, callbacks=callbacks, max_epochs=args.epochs_finetune)
            #if args.resume:
            #    trainer_finetune.fit(model=model, datamodule=dm_target, ckpt_path=ckpt)
            #else:
            trainer_finetune.fit(model=model, datamodule=dm_target)
            try:
                print("Test using best checkpoint !")
                trainer_finetune.test(datamodule=dm_target, ckpt_path="best")
            except:
                print("Error loading best checkpoint ! Test using last checkpoint !")
                trainer_finetune.test(datamodule=dm_target, ckpt_path="last")

        elif args.no_pretrain:
            trainer_finetune = pyl.Trainer(accelerator=accelerator, strategy=strategy, devices=devices, logger=wandb_logger, callbacks=callbacks)
            dm_target.prepare_data()
            dm_target.setup(stage="validate")
            try:
                print("Test using best checkpoint !")
                trainer_finetune.test(model=model, ckpt_path="best", datamodule=dm_target)
            except:
                print("Error loading best checkpoint ! Test using last checkpoint !")
                trainer_finetune.test(model=model, ckpt_path="last", datamodule=dm_target)

if __name__ == "__main__":
    main()




