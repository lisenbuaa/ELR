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
import numpy as np
from torch.nn import init



class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out
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
        self.extern_attention = ExternalAttention(d_model=2048,S=8)
        self.fc = nn.Linear(self.in_channels, num_classes)
        self.gap = torch.nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        output = self.body(x)
        # import pdb
        # pdb.set_trace()
        output2048 = output['3']
        import pdb
        pdb.set_trace()
        features = self.gap(output2048).squeeze()
        output2048 = self.fc(features)

        return output2048, features
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
