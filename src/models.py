from typing import Any, List, Optional, Union
import os

import matplotlib.pyplot as plt

try:
    from MambaCD.changedetection.models.STMambaSCD import STMambaSCD
    from MambaCD.changedetection.configs.config import get_config
except:
    print("No MambaCD : Import error !")

from dchan.scannet import SCanNet, getEncoder

from dchan.scannet_orig import SCanNet as SCanNet_orig, ChangeSimilarity

from . import sepKappa
from scipy import stats as scipy_stats

import wandb
from maskToColor import convert_to_color
import numpy as np
import psutil
import random
import torch.nn.functional as F
import copy

from torchvision.utils import make_grid

from pytorch_lightning.utilities import rank_zero_only

from .changeformer.ChangeFormer import ChangeFormerV6
from PIL import Image
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule, LightningModule, Callback

from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, JaccardLoss
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import (Accuracy, FBetaScore, JaccardIndex, Precision, Recall)
from torchmetrics.wrappers import ClasswiseWrapper

from segmentation_models_pytorch.base import SegmentationModel
def new_forward(self, x, return_features=False, sem_features=None, chg_features=None, concat_features=True, abs_diff=False, encoder_only=False, decoder_only=False):
    if not decoder_only:
        self.check_input_shape(x)
        features = self.encoder(x)
    else:
        features = x
    
    if encoder_only:
        return features
    
    if sem_features is not None:
        sem1, sem2 = sem_features
        for (i,feat) in enumerate(features):
            if i>0:
                features[i] = feat + sem1[i] - sem2[i]
    elif chg_features is not None:
        for (i,feat) in enumerate(features):
            if i>0:
                features[i] = feat + chg_features[i]

    decoder_output = self.decoder(*features)
    
    if self.doSemantics:
        y1 = self.decoderSem(*features)
        if self.double_decoder:
            y2 = self.decoderSem2(*features)
    masks = self.segmentation_head(decoder_output)

    if self.classification_head is not None:
        if self.doSemantics:
            labels = torch.cat((self.classification_head1(y1),self.classification_head2(y1)), dim=1)
        else:
            labels = self.classification_head(decoder_output)
        
        return masks, labels
    
    if return_features:
        return masks, features
    else:
        return masks
    

def computeSepKappa(histKappa, iou, num_classes=20):
    hist_fg = histKappa[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = histKappa[0][0]
    c2hist[0][1] = histKappa.sum(1)[0] - histKappa[0][0]
    c2hist[1][0] = histKappa.sum(0)[0] - histKappa[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = histKappa.copy()
    hist_n0[0][0] = 0
    sc = (np.diag(histKappa[1:,1:]) / (1e-12 + histKappa[1:,1:].sum(axis=0) + histKappa[1:,1:].sum(axis=1))).sum() / (num_classes - 1)
    scs = 0.5*(iou+sc)
    
    kappa_n02 = sepKappa.cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean2 = (iu[0] + iu[1]) / 2
    Sek2 = (kappa_n02 * np.exp(IoU_fg)) / np.e
    pixel_sum = histKappa.sum()
    change_pred_sum  = pixel_sum - histKappa.sum(1)[0].sum()
    change_label_sum = pixel_sum - histKappa.sum(0)[0].sum()
    SC_TP = np.diag(histKappa[1:, 1:]).sum()
    SC_Precision = SC_TP/change_pred_sum
    SC_Recall = SC_TP/change_label_sum
    Fscd = scipy_stats.hmean([SC_Precision, SC_Recall])
    print(f"SC : {sc:.3f}   --   SCS : {scs:.3f}  --  Sek2 : {Sek2:.3f}  --  IoU2 : {IoU_mean2:.3f}  --  Fscd : {Fscd:.3f}")
    return sc, scs, Sek2, IoU_mean2, Fscd



class Lightning(LightningModule):

    def __init__(self, model_name="unet", # model in ["unet", "changeformer", "mambaCD", "scannet"]
                        backbone="resnet50", weights=True, in_channels=3, num_classes=2, loss="ce", lr=0.001, 
                        class_weights: Optional[torch.Tensor] = None,
                        labels: Optional[List[str]] = None,
                        ignore_index: Optional[int] = None,   #### ignore_index pour les classes sémantiques -> mettre 0 pour Flair et HiUCD (unlabeled)
                        doSemantics=False,
                        args=None,
                        out_mapping=None,
                        freeze_encoder=False, freeze_decoder=False):
        super().__init__()

        self.num_classes=num_classes
        if limit_class_test is None:
            limit_class_test = []
        self.restrict_class_test=len(limit_class_test)>0
        self.limit_class_test = limit_class_test
        self.weights = weights
        self.multiclass = (num_classes>2)
        
        self.doSemantics = doSemantics
        self.chg_weight = args.chg_weight
        self.seg_weight = args.seg_weight
       
        
        self.changeSimilarityLoss = args.changeSimilarityLoss
        self.similarityLoss = args.similarityLoss
        self.useSGD = args.sgd
        self.reducedLR = args.reducedLR
        self.map_classes_output = out_mapping is not None
        self.save_preds = args.save_preds
        if self.save_preds:
            self.preds_folder = args.preds_folder
            os.makedirs(f"{args.preds_folder}", exist_ok=True)
                        
        if args.zero_shot and (self.num_classes<10) and (args.mix_dataset=="hiucd"):
            self.num_classes_loss = 10
            self.num_classes = self.num_classes_loss
            self.hparams["num_classes"] = self.num_classes_loss
            self.original_num_classes = num_classes
        else:
            self.num_classes_loss = 0
        if self.map_classes_output:
            print("Doing output mapping for semantic classes !")
            max_key = max(out_mapping.keys())  
            self.out_mapping = torch.zeros(max_key + 1, dtype=torch.long) 
            for key, value in out_mapping.items():
                self.out_mapping[key] = value
            if not args.cpu:
                self.out_mapping = self.out_mapping.to("cuda")

        self.hparams["model_name"] = model_name
        self.save_hyperparameters()

        self.configure_models(args)
    

    def configure_models(self, args):
        model_name: str = self.hparams["model_name"]
        in_channels: int = self.hparams["in_channels"]
        if self.num_classes_loss>0:
            num_classes = self.original_num_classes
        else:
            num_classes: int = self.hparams["num_classes"]
        self.seg_model = None
        if model_name == "unet":
            self.model, self.seg_model = dualUNet.make_model()
        
        elif model_name == "changeformer":
            self.model = ChangeFormerV6(
                input_nc=in_channels,
                output_nc=2,
                decoder_softmax=False,
                embed_dim=256,
            )
        elif model_name=="mambaCD":
            self.model = mambaCD.makeModel(args)
        elif model_name=="scannet":
            self.model = scannet.makeModel(in_channels=3, num_classes=num_classes, input_size=512, resnet_weights_path=args.pretrained_weight_path)
        else:
            raise ValueError(f"Model type '{model_name}' is not valid.")


        # Freeze backbone
        if self.hparams["freeze_encoder"] and model_name in ["unet"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"] and model_name in ["unet"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def configure_finetune(self, only_change=False, freeze_encoder=False, freeze_decoder=False, new_n_classes=0, reset_semantic_head=False, reset_change_head=False):
        if freeze_encoder:
            self.model.encoder.requires_grad_(False)
        if freeze_decoder:
            self.model.decoder.requires_grad_(False)
        #######  SCanNet   ######
        if self.hparams["model_name"]=="scannet":
            if (new_n_classes>0) or (reset_semantic_head):
                new_n_classes = self.hparams["num_classes"] if new_n_classes==0 else new_n_classes
                out, inp, k1, k2 = self.model.classifierA.weight.shape
                self.model.classifierA.weight = torch.nn.Parameter(torch.empty((new_n_classes, inp, k1, k2), device=self.device))
                self.model.classifierA.bias = torch.nn.Parameter(0.1 * torch.ones(new_n_classes, device=self.device))
                self.model.classifierB.weight = torch.nn.Parameter(torch.empty((new_n_classes, inp, k1, k2), device=self.device))
                self.model.classifierB.bias = torch.nn.Parameter(0.1 * torch.ones(new_n_classes, device=self.device))
                self.model.classifierA.reset_parameters()
                self.model.classifierB.reset_parameters()
                
        ###################################

        elif self.hparams["model_name"]=="unet":
            out, inp, k1, k2 = self.seg_model.segmentation_head[0].weight.shape
            self.seg_model.segmentation_head[0].weight = torch.nn.Parameter(torch.empty((new_n_classes, inp, k1, k2), device=self.device))
            torch.nn.init.kaiming_uniform_(self.model.segmentation_head[0].weight)
            self.seg_model.segmentation_head[0].bias = torch.nn.Parameter(0.1 * torch.ones(new_n_classes, device=self.device))
            
            if reset_semantic_head:
                for layer in self.seg_model.segmentation_head.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                        print(abs(layer.weight).mean())
            if reset_change_head:
                for layer in self.model.segmentation_head.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        
        
        self.hparams["num_classes"] = new_n_classes
        self.num_classes = new_n_classes
        self.histKappa = np.zeros((self.num_classes, self.num_classes))
        self.histKappa2 = np.zeros(((self.num_classes-1)**2 + 1, (self.num_classes-1)**2 + 1))
        print("New n_classes : {}".format(self.hparams["num_classes"]))
            
        if new_n_classes>0:
            self.configure_losses()
            self.configure_metrics()

    def forward(self, x):
        if self.hparams["model_name"]=="unet":
            if self.multiclass:
                y1, f1 = self.seg_model(x[:,:x.shape[1]//2], return_features=True)
                y2, f2 = self.seg_model(x[:,x.shape[1]//2:], return_features=True)
                labels = (y1, y2)
                y = self.model(x, sem_features = (f1, f2))
            else:
                y = self.model(x)
                labels = None
            return y, labels
        
        elif self.hparams["model"] in ["mambaCD","changeformer","scannet"]:
            return self.model(x[0],x[1])

    def configure_losses(self):
        loss = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]
        if loss == "ce":
            ignore_value = -1000 if ignore_index is None else ignore_index
            if self.multiclass:
                print(f"Ignore index : {ignore_value} (for semantic loss)")
                self.criterionSem = nn.CrossEntropyLoss(ignore_index=ignore_value, weight=self.hparams["class_weights"])
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
                if self.changeSimilarityLoss:
                    self.changeSimilarity = ChangeSimilarity()

        elif loss == "jaccard":
            self.criterion = JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif loss == "focal":
            self.criterion = FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )
        self.dice = DiceLoss("multiclass", ignore_index=-1)
        print("Losses configured !")


    def configure_metrics(self) -> None:
        num_classes = self.hparams["num_classes"]
        ignore_index = self.hparams["ignore_index"]
        labels = self.hparams["labels"]

        self.f1 = torchmetrics.classification.BinaryF1Score(ignore_index=-1)     #  0 : non-change  /   1 : change  /  -1 : unlabeled
        self.jaccard = torchmetrics.classification.BinaryJaccardIndex(ignore_index=-1)
        self.cfm = {"train":np.zeros(4), "val":np.zeros(4), "test":np.zeros(4)}
        self.histKappa = np.zeros((self.num_classes, self.num_classes))
        self.histKappa2 = np.zeros(((self.num_classes-1)**2 + 1, (self.num_classes-1)**2 + 1))
        self.train_metrics_full = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                ),
                "OverallF1Score": FBetaScore(
                    task="multiclass",
                    num_classes=num_classes,
                    beta=1.0,
                    average="micro",
                    multidim_average="global",
                ),
                "OverallIoU": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "AverageAccuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    multidim_average="global",
                ),
                "AverageF1Score": FBetaScore(
                    task="multiclass",
                    num_classes=num_classes,
                    beta=1.0,
                    average="macro",
                    multidim_average="global",
                ),
                "AverageIoU": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Accuracy": ClasswiseWrapper(
                    Accuracy(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        multidim_average="global",
                    ),
                    labels=labels,
                ),
                "Precision": ClasswiseWrapper(
                    Precision(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        multidim_average="global",
                    ),
                    labels=labels,
                ),
                "Recall": ClasswiseWrapper(
                    Recall(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        multidim_average="global",
                    ),
                    labels=labels,
                ),
                "F1Score": ClasswiseWrapper(
                    FBetaScore(
                        task="multiclass",
                        num_classes=num_classes,
                        beta=1.0,
                        average="none",
                        multidim_average="global",
                    ),
                    labels=labels,
                ),
                "IoU": ClasswiseWrapper(
                    JaccardIndex(
                        task="multiclass", num_classes=num_classes, average="none"
                    ),
                    labels=labels,
                ),
            },
            prefix="train_",
        )

        self.train_metrics = MetricCollection(
                    {
                        "OverallAccuracy": Accuracy(
                            task="multiclass",
                            num_classes=num_classes,
                            average="micro",
                            multidim_average="global",
                        ),
                        "OverallF1Score": FBetaScore(
                            task="multiclass",
                            num_classes=num_classes,
                            beta=1.0,
                            average="micro",
                            multidim_average="global",
                        ),
                        "OverallIoU": JaccardIndex(
                            task="multiclass",
                            num_classes=num_classes,
                            ignore_index=ignore_index,
                            average="micro",
                        ),
                        "AverageAccuracy": Accuracy(
                            task="multiclass",
                            num_classes=num_classes,
                            average="macro",
                            multidim_average="global",
                        ),
                        "AverageF1Score": FBetaScore(
                            task="multiclass",
                            num_classes=num_classes,
                            beta=1.0,
                            average="macro",
                            multidim_average="global",
                        ),
                        "AverageIoU": JaccardIndex(
                            task="multiclass",
                            num_classes=num_classes,
                            ignore_index=ignore_index,
                            average="macro",
                        ),
            },
            prefix="train_",
        )

        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        print("Metrics configured !")


    def configure_optimizers(self):
        lr = self.hparams['lr']
        def lambda_rule(epoch):
            lr_l = 0.99**epoch
            return lr_l
        if self.useSGD:
            print(f"Initial learning rate : {self.hparams['lr']}  ---  Using SGD optimizer")
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], weight_decay=5e-4, momentum=0.9, nesterov=True)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            return {"optimizer": optimizer,"lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},}
        elif self.reducedLR != 1:
            parameters = []
            for model in [self.model, self.seg_model]:
                if model is not None:
                    for name in {name for name, _ in model.named_parameters()}:
                        if name.split(".")[0] in ["encoder","FCN","resCD"]:
                            lr = self.reducedLR * self.hparams['lr']
                            print(f"LR = {lr} for {name}")
                        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                                            'lr':  lr}]
            optimizer = torch.optim.AdamW(parameters, lr=self.hparams["lr"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            return {"optimizer": optimizer,"lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},}
        else:
            print(f"Initial learning rate : {self.hparams['lr']}  ---  Using AdamW optimizer")
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            return {"optimizer": optimizer,"lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},}
        
    def compute_metrics_and_loss(self, y_hat, y, y_sem1=None, y_sem2=None, step="train", test_only_change=False):
        multiclass = (y_sem1 is not None) and (not test_only_change)
        if step=="train":
            metrics_to_log = self.train_metrics
        elif step=="val":
            metrics_to_log = self.val_metrics
        elif step=="test":
            metrics_to_log = self.test_metrics


        ### Binary Change Detection Loss
        loss = self.chg_weight * self.criterion(y_hat, y[:,0])  
        if not multiclass:
            dice_loss = self.dice(y_hat, y[:,0])
            self.log("{}_dice".format(step), dice_loss, on_epoch=True, logger=True, sync_dist=True)
            loss += dice_loss
        ###############################
        
        ### Semantic Change Detection Loss
        if multiclass:
            self.log("{}_loss_bcd".format(step), loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            if y[:,1].sum()>0:  ### if pre-change image has labels
                lossSem1 = self.criterionSem(y_sem1, y[:,1])
                if self.changeSimilarityLoss:
                    similarity_loss = self.changeSimilarity(y_sem1, y_sem2, (y[:,0] > 0).float().unsqueeze(1))
                elif self.similarityLoss:
                    similarity_mask = (y[:,0] == 0).float().unsqueeze(1).expand_as(y_sem1)
                    similarity_loss = F.mse_loss(F.softmax(y_sem1, dim=1) * similarity_mask, F.softmax(y_sem2, dim=1) * similarity_mask, reduction='mean')
                else:
                    similarity_loss = 0
            else:
                lossSem1 = 0
                similarity_loss = 0

            lossSem2 = self.criterionSem(y_sem2, y[:,2])
            self.log("{}_loss_mc1".format(step), lossSem1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log("{}_loss_mc2".format(step), lossSem2, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log("{}_similarityLoss".format(step), similarity_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            loss += self.seg_weight * (lossSem2 + lossSem1) + similarity_loss
        #############################

        self.log("{}_loss".format(step), loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        y_hat = torch.softmax(y_hat, dim=1)
        preds = y_hat.argmax(dim=1)
        if test_only_change & multiclass & self.restrict_class_test:
            preds = preds * ((y_sem1.argmax(dim=1)==self.limit_class_test[0]) | (y_sem2.argmax(dim=1)==self.limit_class_test[0]))
        f1 = self.f1(preds, y[:,0])
        iou = self.jaccard(preds, y[:,0])
        acc = torchmetrics.functional.classification.binary_accuracy(preds, y[:,0], ignore_index=-1)  ###########
        kappa = torchmetrics.functional.classification.binary_cohen_kappa(preds, y[:,0], ignore_index=-1) 
        self.log("{}_f1".format(step), f1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("{}_iou".format(step), iou, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        self.log("{}_accuracy".format(step), acc, on_epoch=True, logger=True, sync_dist=True)
        self.log("{}_kappa".format(step), kappa, on_epoch=True, logger=True, sync_dist=True)

        with torch.no_grad():
            self.cfm[step] += np.array([((preds==1) & (y[:,0]==1)).detach().cpu().numpy().sum(),
                                        ((preds==1) & (y[:,0]==0)).detach().cpu().numpy().sum(),
                                        ((preds==0) & (y[:,0]==1)).detach().cpu().numpy().sum(),
                                        ((preds==0) & (y[:,0]==0)).detach().cpu().numpy().sum(),
                                        ])
        
        if multiclass:
            y_sem1 = torch.softmax(y_sem1, dim=1)
            y_sem1_hard = y_sem1.argmax(dim=1)
            metrics_to_log(y_sem1_hard, y[:,1])  ###########
            self.log_dict({f"{k}_1": v for k, v in metrics_to_log.compute().items()}, sync_dist=True)
            y_sem2 = torch.softmax(y_sem2, dim=1)
            y_sem2_hard = y_sem2.argmax(dim=1)
            metrics_to_log(y_sem2_hard, y[:,2])  ###########
            self.log_dict({f"{k}_2": v for k, v in metrics_to_log.compute().items()}, sync_dist=True)
            if (step=="test") or (step=="val"):
                self.histKappa += sepKappa.get_hist(y_sem1_hard.detach().cpu().numpy(), y[:,1].cpu().numpy(), self.num_classes)
                self.histKappa += sepKappa.get_hist(y_sem2_hard.detach().cpu().numpy(), y[:,2].cpu().numpy(), self.num_classes)
                y_traj = (self.num_classes-1) * (y_sem1_hard-1) + y_sem2_hard
                y_traj[(y_sem1_hard==0) | (y_sem2_hard==0) | (preds==0) | (y_traj<0)] = 0
                y_traj_true = (self.num_classes-1) * (y[:,1]-1) + y[:,2]
                y_traj_true[(y[:,0]==0) | (y_traj_true<0)] = 0
                self.histKappa2 += sepKappa.get_hist(y_traj.detach().cpu().numpy(), y_traj_true.detach().cpu().numpy(), 1 + (self.num_classes-1)**2)
    
        return loss
    

    
    

    def on_train_epoch_end(self):
        tp, fp, fn, tn = self.cfm["train"]
        iou = tp / (tp + fp + fn)
        f1 = 1 / (1 + 0.5 * ( (fp + fn) / tp ))
        self.log_dict({f"train_TP":tp, "train_FP":fp, "train_FN":fn, "train_TN":tn}, sync_dist=True)
        self.cfm["train"] = np.zeros(4)
        self.log("train_F1_total", f1, on_epoch=True, logger=True, prog_bar=True,sync_dist=True)
        self.log("train_IOU_total", iou, on_epoch=True, logger=True, prog_bar=True,sync_dist=True)
       
    def on_validation_epoch_end(self):
        if self.extended_metrics:
            with torch.no_grad():
                self.log_cfm_roc("val")
        tp, fp, fn, tn = self.cfm["val"]
        iou = tp / (tp + fp + fn)
        f1 = 1 / (1 + 0.5 * ( (fp + fn) / tp ))
        self.log_dict({f"val_TP":tp, "val_FP":fp, "val_FN":fn, "val_TN":tn},sync_dist=True)
        self.cfm["val"] = np.zeros(4)
        self.log("val_F1_total", f1, on_epoch=True, logger=True, prog_bar=True,sync_dist=True)
        self.log("val_IOU_total", iou, on_epoch=True, logger=True, prog_bar=True,sync_dist=True)
        if self.multiclass:
            sc, scs, Sek2, IoU_mean2, Fscd = computeSepKappa(self.histKappa2, iou, num_classes=self.num_classes)
            self.log("val_SC", sc, on_epoch=True, logger=True,sync_dist=True)
            self.log("val_SCS", scs, on_epoch=True, logger=True,sync_dist=True)
            self.log("val_sepKappa2", Sek2, on_epoch=True, logger=True,sync_dist=True)
            self.log("val_IoU_sepKappa2", IoU_mean2, on_epoch=True, logger=True,sync_dist=True)
            self.log("val_Fscd", Fscd, on_epoch=True, logger=True,sync_dist=True)
            self.histKappa2 = np.zeros(((self.num_classes-1)**2 + 1, (self.num_classes-1)**2 + 1))

    def on_test_epoch_end(self):
        tp, fp, fn, tn = self.cfm["test"]
        iou = tp / (tp + fp + fn)
        f1 = 1 / (1 + 0.5 * ( (fp + fn) / tp ))
        self.log_dict({f"test_TP":tp, "test_FP":fp, "test_FN":fn, "test_TN":tn},sync_dist=True)
        self.cfm["test"] = np.zeros(4)
        self.log("test_F1_total", f1, on_epoch=True, logger=True, prog_bar=True,sync_dist=True)
        self.log("test_IOU_total", iou, on_epoch=True, logger=True, prog_bar=True,sync_dist=True)
        miou2 = 0.5 * (tp/(tp + fn + fp) + tn/(tn + fp + fn))
        self.log("test_mIOU2", miou2, on_epoch=True, logger=True, prog_bar=True,sync_dist=True)
        if self.multiclass:
            sc, scs, Sek2, IoU_mean2, Fscd = computeSepKappa(self.histKappa2, iou, num_classes=self.num_classes)
            self.log("test_SC", sc, on_epoch=True, logger=True,sync_dist=True)
            self.log("test_SCS", scs, on_epoch=True, logger=True,sync_dist=True)
            self.log("test_sepKappa2", Sek2, on_epoch=True, logger=True,sync_dist=True)
            self.log("test_IoU_sepKappa2", IoU_mean2, on_epoch=True, logger=True,sync_dist=True)
            self.log("test_Fscd", Fscd, on_epoch=True, logger=True,sync_dist=True)
            self.histKappa2 = np.zeros(((self.num_classes-1)**2 + 1, (self.num_classes-1)**2 + 1))

    def training_step(self, batch, batch_idx):
        image1, image2, y = batch["image1"], batch["image2"], batch["mask"]
        model = self.hparams["model"]
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
            y_hat, y_sem = self(x)
            y_sem1, y_sem2 = y_sem if y_sem is not None else (None, None)

        elif model == "changeformer":
            y_hat = self((image1, image2))
            y_sem1, y_sem2 = None, None
        elif model in ["mambaCD", "scannet"]:
            y_hat, y_sem1, y_sem2 = self((image1, image2))
      
        loss = self.compute_metrics_and_loss(y_hat, y, y_sem1, y_sem2, test_only_change=self.train_only_change)

        
        return loss

    def validation_step(self, batch, batch_idx):
        image1, image2, y = batch["image1"], batch["image2"], batch["mask"]
        model = self.hparams["model"]
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
            y_hat, y_sem = self(x)
            y_sem1, y_sem2 = y_sem if y_sem is not None else (None, None)

        elif model == "changeformer":
            y_hat = self((image1, image2))
            y_sem1, y_sem2 = None, None
        elif model in ["mambaCD", "scannet"]:
            y_hat, y_sem1, y_sem2 = self((image1, image2))
        
        loss = self.compute_metrics_and_loss(y_hat, y, y_sem1, y_sem2, step="val", test_only_change=self.train_only_change)
        return loss

        

    def test_step(self, batch, batch_idx):
        image1, image2, y = batch["image1"], batch["image2"], batch["mask"]
        model = self.hparams["model"]
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
            y_hat, y_sem = self(x)
            y_sem1, y_sem2 = y_sem if y_sem is not None else (None, None)

        elif model == "changeformer":
            y_hat = self((image1, image2))
            y_sem1, y_sem2 = None, None
        elif model in ["mambaCD", "scannet"]:
            y_hat, y_sem1, y_sem2 = self((image1, image2))

        if self.save_preds:
            paths = batch["path"]
            for i,chg_preds in enumerate(y_hat):
                y_sem1_hard, y_sem2_hard = y_sem1[i].argmax(dim=0,keepdims=True), y_sem2[i].argmax(dim=0,keepdims=True)
                img_preds = (torch.cat([y_sem1_hard-1, y_sem2_hard -1, y_hat[i].argmax(dim=0,keepdims=True)], dim=0)).cpu().numpy()
                img_preds = np.moveaxis(img_preds, 0, 2).clip(0,255).astype(np.uint8)
                image = Image.fromarray(img_preds,"RGB")
                image.save(f"{self.preds_folder}/{paths[i].split('/')[-1]}")
                
        loss = self.compute_metrics_and_loss(y_hat, y, y_sem1, y_sem2, step="test", test_only_change=self.test_only_change)

        return loss

    

      
        