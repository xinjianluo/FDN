import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import pytorch_ssim
import pickle
from sklearn.cluster import KMeans
import random
import torchvision.utils as tvutils
import numpy as np
import torchvision.models as tvmodels
import RNANcommon  
from AttackModels import ComparativeNN

########################################## Customize Pairs Dataset (Toy Example) ##########################################
 
class PairsDataset(Dataset):

    def __init__(self): 
        copys = 50
        IDStr = "limit16-a50"
        datasetname = "CELEBA"
        self.mixup = torch.load('{}-MixupPublicImages-k6-{}.pt'.format(datasetname, IDStr))
        self.auxi_info = torch.load('{}-public-auxi_info-k6-{}.lst'.format(datasetname, IDStr))
        
        self.copys = copys
        length = int(len(self.mixup)/copys)
        self.totalidx = self.getTotalIdx(length)
         
    def __len__(self):
        return len(self.totalidx) 
    
    def eraseLambda1(self, index):
        img = self.mixup[index]
        info = self.auxi_info[index]   
        return img 
        
    def getSamePairsIdx(self, length):
        same_idx_lst = [i for i in range(length)]
        random.shuffle(same_idx_lst)
        same_paired_idx = torch.zeros(length, 2)
        for i in range(length):
            rd = random.sample(range(self.copys), k=2)
            base_idx = same_idx_lst[i] * self.copys
            same_paired_idx[i][0] = rd[0] + base_idx
            same_paired_idx[i][1] = rd[1] + base_idx
        same_paired_idx = same_paired_idx.int()
        return same_paired_idx

    def getDiffPairsIdx(self, length):
        diff_idx_lst = [i for i in range(length)]
        shu_diff_idx_lst = [i for i in range(length)]
        random.shuffle(shu_diff_idx_lst)
        diff_paired_idx = torch.zeros(length, 2)
        for i in range(length):
            rd = random.sample(range(self.copys), k=2)
            while diff_idx_lst[i] == shu_diff_idx_lst[i]:
                shu_diff_idx_lst[i] = random.randint(0, length-1)
            base_idx_l = diff_idx_lst[i] * self.copys
            base_idx_r = shu_diff_idx_lst[i] * self.copys
            diff_paired_idx[i][0] = rd[0] + base_idx_l
            diff_paired_idx[i][1] = rd[1] + base_idx_r

        diff_paired_idx = diff_paired_idx.int()    
        return diff_paired_idx
        
    def getTotalIdx(self, length):
        same1 = self.getSamePairsIdx(length)
        diff1 = self.getDiffPairsIdx(length)
        same2 = self.getSamePairsIdx(length)
        diff2 = self.getDiffPairsIdx(length)
        same3 = self.getSamePairsIdx(length)
        diff3 = self.getDiffPairsIdx(length)
        cat_idx = torch.cat((same1, same2, same3, diff1, diff2, diff3), dim=0)
        return cat_idx
   
    def __getitem__(self, index):
        pairs_idx = self.totalidx[index] 
        left = pairs_idx[0]
        right = pairs_idx[1]
        x_list = torch.cat( (  self.eraseLambda1(left).unsqueeze(0), self.eraseLambda1(right).unsqueeze(0)  ) )
        label = 1 if index < len(self.totalidx)/2 else 0  
        return x_list, label 

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dm %ds' % (m, s) if h==0 else '%dh %dm %ds' % (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def check_test_accuracy(model, dataloader, accur_base):
    accur = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            yhat = model(*[x[:, i] for i in [0, 1]])
            accur += ( (yhat.argmax(dim=1)) == y ).sum()
    return accur / accur_base

    
if __name__=='__main__':   

    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    # datasets
    copys = 50
    cat_num_image = copys
    catdata = PairsDataset()
    batch_sz = 64
    test_len = int(len(catdata) * 0.2)
    total_len = int(len(catdata))
    train_len = total_len - test_len
    trainset, testset = torch.utils.data.random_split(catdata, [train_len, test_len])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sz, shuffle=True)
    print("len(trainset):", len(trainset), "len(testset)", len(testset))
    print("len(trainloader):", len(trainloader), "len(testloader):", len(testloader))
    
    # train model 
    losses = []
    train_accur = []
    test_accur = []

    compnn = ComparativeNN().to(device)
    compnn_optimizer = torch.optim.Adam(compnn.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    start = time.time()
    epochs = 50

    test_accur_base = len(testset) 
    test_interval = len(trainloader)-10

    for epoch in range(1, epochs+1): 
        accurate = 0.0
        train_accur_base = 0.0
        for x, y in trainloader:
            compnn.train()
            x = x.to(device)
            y = y.to(device)
            compnn_optimizer.zero_grad()
            yhat = compnn(*[x[:, i] for i in [0, 1]])
            loss = loss_fn(yhat, y)
            loss.backward()
            
            compnn_optimizer.step()
            accurate += ( (yhat.argmax(dim=1))==y).sum()
            train_accur_base += x.shape[0]
            
        losses.append(loss)
        train = accurate / train_accur_base
        train_accur.append(train)
        test = check_test_accuracy(compnn, testloader, test_accur_base) 
        test_accur.append(test)
        print("In epoch {}/{}, train accur. is {}, test accur. is {}.".format(epoch, epochs, train, test)) 
                
        print('%s (%d %d%%)' % (timeSince(start, epoch / epochs), epoch, epoch / epochs * 100))
        print("-----------------------------")
             
    torch.save(compnn.state_dict(), "ComparativeNetwork-{}ep".format(epochs))
    print("The ComparativeNN model has been saved!")




    
   