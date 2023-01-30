import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
import torch.optim as optim
import numpy as np

class BaseEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(simCLR, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
    
class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        self.f = BaseEncoder().f
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

class simCLR:
    def __init__(self, config):
        self.lr = config['lr']
        self.temp = config['temperature']
        self.epoch = config['epoch']
        self.batch_size = config['batch_size']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model  = BaseEncoder(config['num_features'])
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
    
    def simclr_loss(self,out_left, out_right):
        N = out_left.shape[0]    
        out = torch.cat([out_left, out_right], dim=0)
        
        out = out/torch.linalg.norm(out,dim=1,keepdim=True)
        sim_matrix = out@out.transpose(0,1)
        
        exponential = (sim_matrix/self.temp).exp().to(self.device)
        mask = (torch.ones_like(exponential, device=self.device) - torch.eye(2 * N, device=self.device)).to(self.device).bool()
        exponential = exponential.masked_select(mask).view(2 * N, -1)
        denom = exponential.sum(dim=1,keepdims=True).to(self.device)
        
        l= out_left/torch.linalg.norm(out_left,dim=1,keepdim=True)
        r = out_right/torch.linalg.norm(out_right,dim=1,keepdim=True)
        x = torch.sum(l*r,dim=1,keepdim=True)
        x = torch.cat([x,x],dim=0).to(self.device)
        numerator = (x/self.tau).exp().to(self.device)
        
        return torch.sum(-torch.log(numerator / denom) / (2*N)) 
    
    def train_one_epoch(self,train_data):
        self.model.train()
        epoch_loss = []
        
        for data in train_data:
            x_i, x_j, target = data
            x_i, x_j = x_i.to(self.device), x_j.to(self.device)

            _,out_left = self.model(x_i)
            _,out_right = self.model(x_j)
            loss = self.simclr_loss(out_left,out_right, self.temp, device=self.device)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            epoch_loss.append(loss.item())
            
        return np.mean(epoch_loss)
    
    def train(self,train_data):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            self.model.train(True)
            loss = self.train_one_epoch(train_data) 

            if epoch%100==0:
                print(f'Epoch {epoch+1}/{self.epochs}:- Loss: {loss:.2f}')
    
    def test():
        pass
    
    def save_model(self,file_name):
        torch.save(self.model.state_dict(),file_name)
    
    def load_model(self,file_name):
        self.model.load_state_dict(torch.load(file_name))