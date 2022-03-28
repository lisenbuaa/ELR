import torch.nn.functional as F
import torch
import numpy as np
from parse_config import ConfigParser
import torch.nn as nn
from torch.autograd import Variable
import math
from utils import sigmoid_rampup, sigmoid_rampdown, cosine_rampup, cosine_rampdown, linear_rampup
import torch.nn as nn
from scipy.stats import ortho_group

m = ortho_group.rvs(dim=128)


def cross_entropy(output, target, M=3):
    return F.cross_entropy(output, target)

class elr_plus_loss(nn.Module):
    def __init__(self, num_examp, config, device, num_classes=10, beta=0.3):
        super(elr_plus_loss, self).__init__()
        self.config = config
        self.pred_hist = (torch.zeros(num_examp, num_classes)).to(device)
        self.train_clean = (torch.ones(num_examp,)).to(device)
        self.q = 0
        self.beta = beta
        self.num_classes = num_classes
        self.mse = nn.MSELoss()
        # one_vector = torch.ones(self.num_classes, 128).cuda()
        # self.memeory_ut = torch.div(one_vector,torch.norm(one_vector))
        self.memeory_ut = torch.tensor(np.float32(m[:num_classes,:])).cuda()

    def forward(self, iteration, output, y_labeled,gt_label, vt,features_highdim,features_resconstruct,epoch):
        vt = vt.squeeze()
        y_pred = F.softmax(output,dim=1)

        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled*self.q
            y_labeled = y_labeled/(y_labeled).sum(dim=1,keepdim=True)

        # import pdb
        # pdb.set_trace()
        if epoch > 2:
            y_pred_score = y_pred.gather(1,gt_label.unsqueeze(1))
            clean_index = y_pred_score >= 0.5
            noise_index = y_pred_score < 0.5
            clean_index = clean_index.squeeze()
            noise_index = noise_index.squeeze()
            # import pdb
            # pdb.set_trace()
            clean_ce_loss = -torch.sum(y_labeled * F.log_softmax(output, dim=1), dim = -1) 
            noise_ce_loss = -torch.sum(self.q * F.log_softmax(output, dim=1), dim = -1)
            ce_loss = torch.mean(clean_ce_loss[clean_index])+ torch.mean(noise_ce_loss[noise_index])

        else:
            ce_loss = torch.mean(-torch.sum(y_labeled * F.log_softmax(output, dim=1), dim = -1))

    
        elr_reg = ((1-(self.q * y_pred).sum(dim=1)).log()).mean()
        # final_loss = ce_loss + sigmoid_rampup(iteration, self.config['coef_step'])*(self.config['train_loss']['args']['lambda']*reg)
        # weight = self.q.detach()
        weight = y_pred
        #### add by lisen
        features_loss = self.mse(torch.torch.mm(weight,self.memeory_ut), vt)
        reconstruct_loss = self.mse(features_highdim, features_resconstruct)

        final_loss = ce_loss + sigmoid_rampup(iteration, self.config['coef_step'])*(self.config['train_loss']['args']['lambda'])*elr_reg + sigmoid_rampup(iteration, self.config['coef_step'])*(self.config['train_loss']['args']['gamma'])*features_loss + sigmoid_rampup(iteration, self.config['coef_step'])*(self.config['train_loss']['args']['alpha'])*reconstruct_loss
        
        # weight_norm = torch.norm(weight)

        # v_parallel = torch.mm(weight,self.memeory_ut) 
        # v_parallel_norm = torch.norm(v_parallel,dim = 1)
        # v_parallel_norm = v_parallel_norm.repeat(1,v_parallel.shape[1])

        
        # v_vertical = vt - v_parallel
        # v_vertical_norm = torch.norm(v_vertical,dim = 1).unsqueeze(1)
        # v_parallel_norm = v_vertical_norm.repeat(1,v_vertical.shape[1])

        # thegma = v_parallel_norm * v_parallel_norm

        # self.memeory_ut = self.memeory_ut + \
        # torch.matmul(torch.div(weight,weight_norm).transpose(1,0),(torch.cos(thegma*self.n_size)-1)*torch.div(v_parallel,v_parallel_norm) + \
        # torch.sin(thegma*self.n_size)*torch.div(v_vertical,v_vertical_norm))
        # self.memeory_ut = self.memeory_ut.detach()
      
        return  final_loss, y_pred.cpu().detach()

    # def update_hist(self, epoch, out, feature_lowdim, index= None, mix_index = ..., mixup_l = 1):
    def update_hist(self, epoch, out,  feature_lowdim, index):
        self.n_size = 1 / epoch
        y_pred_ = F.softmax(out,dim=1)

        weight = y_pred_.detach()

        #### add by lisen
        
        weight_norm = torch.norm(weight)

        v_parallel = torch.mm(weight,self.memeory_ut) 
        v_parallel_norm = torch.norm(v_parallel,dim = 1)
        v_parallel_norm = v_parallel_norm.repeat(1,v_parallel.shape[1])

        vt = feature_lowdim.squeeze()
        v_vertical = vt - v_parallel
        v_vertical_norm = torch.norm(v_vertical,dim = 1).unsqueeze(1)
        v_parallel_norm = v_vertical_norm.repeat(1,v_vertical.shape[1])

        thegma = v_parallel_norm * v_parallel_norm

        self.memeory_ut = self.memeory_ut + \
        torch.matmul(torch.div(weight,weight_norm).transpose(1,0),(torch.cos(thegma*self.n_size)-1)*torch.div(v_parallel,v_parallel_norm) + \
        torch.sin(thegma*self.n_size)*torch.div(v_vertical,v_vertical_norm))
        self.memeory_ut = self.memeory_ut.detach()


        if epoch > 2:
        ##### pred the softlabel by grouse
        # import pdb
        # pdb.set_trace()
            temp = self.memeory_ut.transpose(1,0)
            y_pred_grouse = torch.mm(feature_lowdim, temp)
            tau = 0.5
            y_pred_grouse = F.softmax(y_pred_grouse/tau,dim=1)
            self.pred_hist[index] = self.beta * self.pred_hist[index] +  (1-self.beta) *  y_pred_grouse/(y_pred_grouse).sum(dim=1,keepdim=True)
            self.q = self.pred_hist[index]
        else:
            self.pred_hist[index] = self.beta * self.pred_hist[index] +  (1-self.beta) *  y_pred_/(y_pred_).sum(dim=1,keepdim=True)


            


            

        
