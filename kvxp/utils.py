# -*- coding: utf-8 -*-
"""
Author
------
JDL
Email
-----
jiadong.li at nyu.edu
Created on
----------
- Fri Jan 31 12:00:00 2023
Modifications
-------------
- Fri Feb 14 12:00:00 2023
Aims
----
- tool script
"""
import torch
import torch.nn as nn

def chi(pred, tgt, e_pred, e_tgt):
    var = torch.sqrt(torch.square(e_pred)+torch.square(e_tgt))
    return (pred-tgt)/var

def kl_divergence(mean_pred, std_pred, mean2=0, std2=1):
    var_pred = torch.square(std_pred)
    # var2 =  torch.square(std2)
    var2 = std2**2
    return 0.5*(torch.log(var2/var_pred) + (var_pred + (mean_pred - mean2)**2)/var2 - 1.)


def cost_penalty(pred, tgt, e_pred, e_tgt, model, lmbda=1e-10, lmbda_2=1):
    """
    Computes the loss between the predictions and targets, with a penalty term.
    """
    err_min, err_max = 1e-4, 20

    chi_pred = chi(pred, tgt, e_pred, e_tgt)
    kl_pen = lmbda_2*kl_divergence(chi_pred.mean(axis=0), chi_pred.std(axis=0)).sum()

    cost = torch.nn.GaussianNLLLoss(reduction='mean', eps=err_min)
    # jitter = f_jitter * tgt
    var = torch.sqrt(torch.square(e_pred)+torch.square(e_tgt))
    loss = cost(pred, tgt, var)

    loss +=  kl_pen
    return loss

def cost_val(pred, tgt, e_pred, e_tgt):
    err_min, err_max = 1e-4, 200

    var = torch.sqrt(torch.square(e_pred)+torch.square(e_tgt))
    cost = torch.nn.GaussianNLLLoss(reduction='mean', eps=err_min)
    loss = cost(pred, tgt, var)
    return loss

def cost_mse(pred, tgt):
    cost = torch.nn.MSELoss(reduction='mean')
    return cost(pred, tgt)


def cost_mse_pen(pred, tgt, model, lbda=1e-5):
    cost = torch.nn.MSELoss(reduction='mean')
    loss = cost(pred, tgt)
    l2_reg = torch.tensor(0.0)

    for param in model.parameters():
        l2_reg += torch.norm(param).cpu()
    loss += lbda * l2_reg
    return loss


class WeightedMSE(nn.Module):
    def __init__(self, pos_weight, thresold=-0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.thresold   = thresold
    
    def forward(self, inputs, targets):
        mse_loss = nn.functional.mse_loss(inputs, targets)
        weights = torch.ones_like(targets)
        # weights[targets[:,-1]<0.3] = 1.0
        # weights[targets[:,-1]>=0.3] = self.pos_weight
        weights[targets[:,2]<self.thresold] = 1.0
        weights[targets[:,2]>self.thresold] = self.pos_weight
        weighted_loss = weights * mse_loss
        return torch.mean(weighted_loss)


def remove_prefix(state_dict, unwanted_prefix = '_orig_mod.'):
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

