

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random
import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)


class SphereProduct2(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, lamb=0.7, r=30, m=0.4, t=3, b=0.25):
        super(SphereProduct2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lamb = lamb
        self.r = r
        self.m = m
        self.t = t
        self.b = b
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: 2 * ((x + 1) / 2) ** self.t - 1,
        ]

    def forward(self, input, label):

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.r * (self.mlambda[0](cos_theta) - self.m) + self.b
        cos_m_theta1 = self.r * (self.mlambda[0](cos_theta) + self.m) + self.b
        cos_p_theta = (self.lamb / self.r) * torch.log(1 + torch.exp(-cos_m_theta))

        cos_n_theta = ((1 - self.lamb) / self.r) * torch.log(1 + torch.exp(cos_m_theta1))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Calculate output ---------------------------
        loss = (one_hot * cos_p_theta) + (1 - one_hot) * cos_n_theta
        loss = loss.sum(dim=1)
        
        return loss.mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'



