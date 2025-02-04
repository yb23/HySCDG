from arch.scannet_orig import SCanNet, ChangeSimilarity


def makeModel(in_channels=3, num_classes=20, input_size=512, resnet_weights_path=""):
    return SCanNet(in_channels=in_channels, num_classes=num_classes, input_size=input_size, resnet_weights_path=resnet_weights_path)
