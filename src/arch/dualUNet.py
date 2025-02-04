
import torch
import torch.nn as nn
import copy
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
    
SegmentationModel.forward = new_forward
import segmentation_models_pytorch as smp


def makeModel(multiclass=True, backbone="resnet", pretrained=True, in_channels=3, num_classes=20):
    if multiclass:
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels * 2,  # images are concatenated
            classes=2,)
        model.classification_head = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                                                        nn.ReLU(),
                                                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                                                        nn.ReLU(),
                                                                        nn.Conv2d(in_channels=32, out_channels=num_classes*2, kernel_size=3, padding=1),
                                                                        nn.Sigmoid()
        )
        model.doSemantics = True
        seg_model = smp.Unet(encoder_name=backbone, encoder_weights="imagenet" if pretrained else None, in_channels=in_channels, classes=num_classes,)

    else:
        model = smp.Unet(encoder_name=backbone,encoder_weights="imagenet" if pretrained is True else None, in_channels=in_channels*2, classes=2)
        seg_model = None

    return model, seg_model
