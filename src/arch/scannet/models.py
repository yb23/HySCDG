import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from .CSWin_Transformer import mit


class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0.1, reduction=reduction)
        
    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = torch.reshape(x1,[b*h*w,c])
        x2 = torch.reshape(x2,[b*h*w,c])
        
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target,[b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss
    

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


args = {'hidden_size': 128*3,
        'mlp_dim': 256*3,
        'num_heads': 4,
        'num_layers': 2,
        'dropout_rate': 0.}


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low//scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low//scale_ratio),
            nn.BatchNorm2d(in_channels_low//scale_ratio),
            nn.ReLU(inplace=True) )
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) )

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x

class FCN(nn.Module):
    def __init__(self, in_channels=3, resnet_weights_path="resnet34.pth", pretrained=True):
        super(FCN, self).__init__()
        try:
            resnet = models.resnet34(pretrained=pretrained)
        except:
            #### Ne marche pas a priori : checkpoint incompatible
            checkpoint_resnet = torch.load(resnet_weights_path)
            resnet.load_state_dict(checkpoint_resnet)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SCanNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, input_size=512, resnet_weights_path="resnet34.pth"):
        super(SCanNet, self).__init__()
        feat_size = input_size//4
        self.FCN = FCN(in_channels, pretrained=True, resnet_weights_path=resnet_weights_path)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.transformer = mit(img_size=feat_size, in_chans=128*3, embed_dim=128*3)
        
        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1  = _DecoderBlock(128, 64,  128)
        self.Dec2  = _DecoderBlock(128, 64,  128)
        
        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
        
        ################## Modification to get 2 output layers ##############################
        #self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 2, kernel_size=1))

        initialize_weights(self.Dec1, self.Dec2, self.classifierA, self.classifierB, self.resCD, self.DecCD, self.classifierCD)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        
        x = self.FCN.layer0(x) #size:1/2
        x = self.FCN.maxpool(x) #size:1/4
        x_low = self.FCN.layer1(x) #size:1/4
        x = self.FCN.layer2(x_low) #size:1/8
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        return x, x_low
    
    def CD_forward(self, x1, x2):
        b,c,h,w = x1.size()
        x = torch.cat([x1,x2], 1)
        xc = self.resCD(x)
        return x1, x2, xc
    
    def forward(self, x1, x2):
        x_size = x1.size()
        
        x1, x1_low = self.base_forward(x1)
        x2, x2_low = self.base_forward(x2)
        x1, x2, xc = self.CD_forward(x1, x2)

        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)        
        xc_low = torch.cat([x1_low, x2_low], 1)
        xc = self.DecCD(xc, xc_low)
                
        x = torch.cat([x1, x2, xc], 1)
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]
        
        out1 = self.classifierA(x1)
        out2 = self.classifierB(x2)
        change = self.classifierCD(xc)

        change = F.upsample(change, x_size[2:], mode='bilinear')

        #####  MODIFICATION : Pour avoir 2 couches en sortie
        change = torch.sigmoid(change)
        change = torch.cat((1-change, change), dim=1)
        
        return change, F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear')