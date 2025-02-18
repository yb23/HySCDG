import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb

from torchvision.utils import make_grid
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import Callback
from .utils import convert_to_color

class ImagePredictionLoggerSemantic(Callback):
    def __init__(self, val_samples, num_samples=8, log_every_n_epochs=10, name="examples", class_mapping=None, norm_weights=None):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgsA, self.val_imgsB, self.val_labels = val_samples["image1"], val_samples["image2"], val_samples["mask"]
        self.log_every_n_epochs = log_every_n_epochs
        self.name = name
        if class_mapping is not None:
            self.need_class_mapping = True
            self.getDict = lambda x:class_mapping.get(x, 0)
        else:
            self.need_class_mapping = False
        if norm_weights is not None:
            self.do_denorm = True
            (self.mA, self.sA), (self.mB, self.sB) = norm_weights
            self.imgsA = (val_samples["image1"] * self.sA[None,:,None,None] + self.mA[None,:,None,None])/255
            self.imgsB = (val_samples["image2"] * self.sB[None,:,None,None] + self.mB[None,:,None,None])/255
        else:
            self.do_denorm = False
            self.imgsA, self.imgsB = val_samples["image1"], val_samples["image2"]
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % self.log_every_n_epochs == 0:
            with torch.no_grad():
                if pl_module.hparams["model_name"] in ["mambaCD","scannet"]:
                    y_hat, y_sem1, y_sem2 = pl_module((self.val_imgsA.to(device=pl_module.device), self.val_imgsB.to(device=pl_module.device)))
                else:
                    y_hat, y_sem = pl_module((self.val_imgsA.to(device=pl_module.device), self.val_imgsB.to(device=pl_module.device)))
                    y_sem1, y_sem2 = y_sem
                logits, logits1, logits2 = torch.softmax(y_hat, dim=1), torch.softmax(y_sem1, dim=1), torch.softmax(y_sem2, dim=1)
                preds = logits.argmax(dim=1)
                preds1 = logits1.argmax(dim=1).cpu().numpy()
                preds2 = logits2.argmax(dim=1).cpu().numpy()
            if self.need_class_mapping:
                preds1 = np.vectorize(self.getDict)(preds1)
                preds2 = np.vectorize(self.getDict)(preds2)
                lbls = np.vectorize(self.getDict)(self.val_labels)
            else:
                lbls = self.val_labels
            wandb_Images = []
            
            for inum in range(self.num_samples):
                fig, axs = plt.subplots(3,3,figsize=(10,10))
                axs[0,0].imshow(np.moveaxis(self.imgsA[inum][:3].cpu().numpy(),0,2))
                axs[0,1].imshow(np.moveaxis(self.imgsB[inum][:3].cpu().numpy(),0,2))
                axs[0,2].imshow(convert_to_color(self.val_labels[inum][0],palette={-1:"#FFFFFF", 0:'#000000',1:'#FFFFFF'}))#palette={0:'#FF00FF', 1:'#000000',2:'#FFFFFF'}))
                axs[1,0].imshow(convert_to_color(preds1[inum]))
                axs[1,1].imshow(convert_to_color(preds2[inum]))
                axs[1,2].imshow(convert_to_color(preds[inum].cpu().numpy(),palette={0:'#000000',1:'#FFFFFF'}))
                if len(lbls[inum])>1:
                    axs[2,0].imshow(convert_to_color(lbls[inum][1]))
                    axs[2,1].imshow(convert_to_color(lbls[inum][2]))
                axs[2,2].imshow(logits[inum][1].cpu().numpy())
                wandb_Images.append(wandb.Image(fig))
                plt.clf()
                plt.close()

            # Log the images as wandb Image
            trainer.logger.experiment.log({self.name:wandb_Images})



class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=16, log_every_n_epochs=10, name="examples_binary"):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgsA, self.val_imgsB, self.val_labels = val_samples["image1"], val_samples["image2"], val_samples["mask"]
        self.log_every_n_epochs = log_every_n_epochs
        self.name = name

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % self.log_every_n_epochs == 0:
            # Bring the tensors to CPU
            val_imgsA = self.val_imgsA.to(device=pl_module.device)
            val_imgsB = self.val_imgsB.to(device=pl_module.device)
            val_labels = self.val_labels.to(device=pl_module.device)
            # Get model prediction
            with torch.no_grad():
                if pl_module.hparams["model_name"] in ["mambaCD","scannet"]:
                    y_hat, y_sem1, y_sem2 = pl_module([val_imgsA, val_imgsB])
                else:
                    y_hat, y_sem = pl_module((val_imgsA, val_imgsB))
                    y_sem1 = y_sem, y_sem2 if y_sem is not None else (None, None)
                
                logits = torch.softmax(y_hat, dim=1)
                logits = logits[:,[1]].repeat(1,3,1,1)
                if pl_module.restrict_class_test:
                    is_good_class = ((y_sem1.argmax(dim=1)==pl_module.limit_class_test[0]) | (y_sem2.argmax(dim=1)==pl_module.limit_class_test[0]))
                    logits[:,1] *= is_good_class
                    logits[:,2] *= ~is_good_class
                val_labels = val_labels[:,[0]].repeat(1,3,1,1)
                val_labels[val_labels==-1] = 0.5
                trainer.logger.experiment.log({self.name:[wandb.Image(make_grid([val_imgsA[i,:3], val_imgsB[i,:3], logits[i], val_labels[i]], nrow=2)) for i in range(self.num_samples)]})
