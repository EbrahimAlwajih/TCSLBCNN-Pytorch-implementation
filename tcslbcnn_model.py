import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvTCSLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5,threshold=[1, 0.5,0.75, 0.25]):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0)
              
        self.nInputPlane = in_channels
        self.nOutputPlane = out_channels
        self.kW = kernel_size
        matrix_proba=torch.reshape(matrix_proba,(self.nOutputPlane,self.nInputPlane,self.kW*self.kW))
        binary_weights = matrix_proba
        index1=np.array([0,1,2,3,5,6,7,8])
       
        i=1
        for nInputPlane in range(1,self.nInputPlane):
            np.random.shuffle(index1)
            for nOutputPlane in range( 1,self.nOutputPlane):
                threshold_idx=np.random.randint(0,len(threshold))
                rand1=np.random.randint(1,8)
                binary_weights[nOutputPlane,nInputPlane,index1[rand1]]=threshold[threshold_idx]
                binary_weights[nOutputPlane,nInputPlane,8-index1[rand1]]=-threshold[threshold_idx]
                i=i+1
        
        
        binary_weights = binary_weights.view(self.nOutputPlane,self.nInputPlane,self.kW,self.kW)
       # self.CSLBCNN.weight = Parameter(self.binary_weights)
        
        
      
        weights.data = binary_weights
        #print (weights.data)
        weights.requires_grad_(False)


class TCSLBPBlock(nn.Module):

    def __init__(self, numChannels, numWeights, sparsity=0.5):
        super().__init__()
        self.BATCH_NORM = nn.BatchNorm2d(numChannels)
        self.TCSLBCNN_LAYER = ConvTCSLBP(numChannels, numWeights, kernel_size=3, sparsity=sparsity)
        self.RELU = nn.ReLU()
        self.CONV_1X1 = nn.Conv2d(numWeights, numChannels, kernel_size=1)
        
    def forward(self, x):
        
        residual = x
        x = self.BATCH_NORM(x)
        x = self.TCSLBCNN_LAYER(x)
        x = self.RELU(x)
        x = self.CONV_1X1(x)
        x.add_(residual)
        return x


class TCSLBCNN(nn.Module):
    def __init__(self, nInputPlane=1, numChannels=16, numWeights=256, full=50, depth=2, sparsity=0.5):
        super().__init__()

        self.preprocess_block = nn.Sequential(
            nn.Conv2d(nInputPlane, numChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(numChannels),
            nn.ReLU(inplace=True)
        )

        chain = [TCSLBPBlock(numChannels, numWeights, sparsity) for i in range(depth)]
        self.chained_blocks = nn.Sequential(*chain)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=5)
        self.RELU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(numChannels * 5 * 5, full)   # for mnist (image size is different 28*28)
        #self.fc1 = nn.Linear(numChannels * 6 * 6, full) # for cifar 
        self.fc2 = nn.Linear(full, 10)

    def forward(self, x):
        x = self.preprocess_block(x)
        x = self.chained_blocks(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.RELU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
