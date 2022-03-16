import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .ResNet_Zoo import ResNet, BasicBlock, Bottleneck
from .PreResNet import PreActResNet, PreActBlock
import torchvision.models as models
from .InceptionResNetV2 import InceptionResNetV2
import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    #return models.resnet34(num_classes=10)

class resnet50(torch.nn.Module):
    def __init__(self,num_classes = 14):
        super(resnet50, self).__init__()
        self.in_channels = 2048
        import torchvision.models as models
        self.model_ft = models.resnet50(pretrained=True)
        self.body = create_feature_extractor(
            self.model_ft, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        self.fc = nn.Linear(self.in_channels, num_classes)
        self.fc_low_dim = nn.Linear(self.in_channels, 20)
        self.gap = torch.nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        output = self.body(x)
        # import pdb
        # pdb.set_trace()
        output2048 = output['3']
        features = self.gap(output2048).squeeze()
        features_lowdim = self.fc_low_dim(features)
        output2048 = self.fc(features)

        return output2048, features_lowdim
# def resnet50(num_classes=14):
#     import torchvision.models as models
#     model_ft = models.resnet50(pretrained=True)
#     # import pdb
#     # pdb.set_trace()
#     num_ftrs = model_ft.fc.in_features
#     model_ft.fc = nn.Linear(num_ftrs, num_classes)
#     return model_ft


def PreActResNet34(num_classes=10) -> PreActResNet:
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes)
def PreActResNet18(num_classes=10) -> PreActResNet:
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)
