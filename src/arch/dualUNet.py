
import torch
import torch.nn as nn
import copy
from segmentation_models_pytorch.base import SegmentationModel
def new_forward(self, x, return_features=False, sem_features=None, chg_features=None, encoder_only=False, decoder_only=False):
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
    
    masks = self.segmentation_head(decoder_output)
    
    if return_features:
        return masks, features
    else:
        return masks
    

SegmentationModel.forward = new_forward
import segmentation_models_pytorch as smp


def makeModel(multiclass=True, backbone="resnet50", pretrained=True, in_channels=3, num_classes=20):
    model = smp.Unet(encoder_name=backbone, encoder_weights="imagenet" if pretrained else None, in_channels=in_channels * 2, classes=2,)
    
    if multiclass:
        seg_model = smp.Unet(encoder_name=backbone, encoder_weights="imagenet" if pretrained else None, in_channels=in_channels, classes=num_classes,)
    else:
        seg_model = None

    return model, seg_model
