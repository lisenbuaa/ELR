import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .ResNet_Zoo import ResNet, BasicBlock



def resnet34(num_classes=10):
    model = ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    # import pdb
    # pdb.set_trace()
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
