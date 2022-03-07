import torch.nn.functional as F
import torch
from parse_config import ConfigParser
import torch.nn as nn


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = beta
        one_vector = torch.ones(self.num_classes,512).cuda()
        self.memeory_ut = torch.div(one_vector,torch.norm(one_vector))
        
        self.mse = nn.MSELoss()
        

    def forward(self, index, output, label, vt, epoch):
        self.n_size = 1/epoch
        vt = vt.squeeze()
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()

        # import pdb
        # pdb.set_trace()

        weight = self.target[index].detach()
        #### add by lisen
        features_loss = self.mse(torch.torch.mm(weight,self.memeory_ut), vt)

        final_loss = ce_loss +  self.config['train_loss']['args']['lambda']*elr_reg + features_loss

        
        weight_norm = torch.norm(weight)

        v_parallel = torch.mm(weight,self.memeory_ut) 
        v_parallel_norm = torch.norm(v_parallel,dim = 1)
        v_parallel_norm = v_parallel_norm.repeat(1,v_parallel.shape[1])

        
        v_vertical = vt - v_parallel
        v_vertical_norm = torch.norm(v_vertical,dim = 1).unsqueeze(1)
        v_parallel_norm = v_vertical_norm.repeat(1,v_vertical.shape[1])

        thegma = v_parallel_norm * v_parallel_norm

        self.memeory_ut = self.memeory_ut + \
        torch.matmul(torch.div(weight,weight_norm).transpose(1,0),(torch.cos(thegma*self.n_size)-1)*torch.div(v_parallel,v_parallel_norm) + \
        torch.sin(thegma*self.n_size)*torch.div(v_vertical,v_vertical_norm))
        self.memeory_ut = self.memeory_ut.detach()
        # import pdb
        # pdb.set_trace()

        return  final_loss

