# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:18:49 2021

@author: angelou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.conv_layer import Conv, BNPReLU
import math

class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3,dkSize=3):
        super().__init__()
        
        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)
        
        self.dconv_4_1 = Conv(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (1*d+1,1*d+1),
                            dilation=(d+1,d+1), groups = nIn //16, bn_acti=True)
        
        self.dconv_4_2 = Conv(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (1*d+1,1*d+1),
                            dilation=(d+1,d+1), groups = nIn //16, bn_acti=True)
        
        self.dconv_4_3 = Conv(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (1*d+1,1*d+1),
                            dilation=(d+1,d+1), groups = nIn //16, bn_acti=True)
        
        
        
        self.dconv_1_1 = Conv(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (1,1),
                            dilation=(1,1), groups = nIn //16, bn_acti=True)
        
        self.dconv_1_2 = Conv(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (1,1),
                            dilation=(1,1), groups = nIn //16, bn_acti=True)
        
        self.dconv_1_3 = Conv(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (1,1),
                            dilation=(1,1), groups = nIn //16, bn_acti=True)
        
        
        
        self.dconv_2_1 = Conv(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (int(d/4+1),int(d/4+1)),
                            dilation=(int(d/4+1),int(d/4+1)), groups = nIn //16, bn_acti=True)
        
        self.dconv_2_2 = Conv(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (int(d/4+1),int(d/4+1)),
                            dilation=(int(d/4+1),int(d/4+1)), groups = nIn //16, bn_acti=True)
        
        self.dconv_2_3 = Conv(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (int(d/4+1),int(d/4+1)),
                            dilation=(int(d/4+1),int(d/4+1)), groups = nIn //16, bn_acti=True)
        
        
        self.dconv_3_1 = Conv(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (int(d/2+1),int(d/2+1)),
                            dilation=(int(d/2+1),int(d/2+1)), groups = nIn //16, bn_acti=True)
        
        self.dconv_3_2 = Conv(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (int(d/2+1),int(d/2+1)),
                            dilation=(int(d/2+1),int(d/2+1)), groups = nIn //16, bn_acti=True)
        
        self.dconv_3_3 = Conv(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (int(d/2+1),int(d/2+1)),
                            dilation=(int(d/2+1),int(d/2+1)), groups = nIn //16, bn_acti=True)
        
                      
        
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0,bn_acti=False)  
        
    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)
        
        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)
        
        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)
        
        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)
        
        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)
        
        output_1 = torch.cat([o1_1,o1_2,o1_3], 1)
        output_2 = torch.cat([o2_1,o2_2,o2_3], 1)      
        output_3 = torch.cat([o3_1,o3_2,o3_3], 1)       
        output_4 = torch.cat([o4_1,o4_2,o4_3], 1)   
        
        
        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1,ad2,ad3,ad4],1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        
        return output+input