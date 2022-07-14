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
from AttackModels import FusionNet, RNAN

########################################## DataSets ##########################################
class ClusterDataset(Dataset):

    def __init__(self, cat_img_num, copys, limit, datasetname, parameterK, suffix): 
        IDStr = "limit{}-a{}".format(limit, copys)
        rootdir = ""
        self.mixup = torch.load('{}{}-MixupPublicImages-k{}-{}{}.pt'.format(rootdir, datasetname, parameterK, IDStr, suffix))
        self.auxi_info = torch.load('{}{}-public-auxi_info-k{}-{}{}.lst'.format(rootdir, datasetname, parameterK, IDStr, suffix))  
        self.augmented_img = torch.load('{}{}-AugmentatedImages-{}.pt'.format(rootdir, datasetname, IDStr)) 
        
        self.cat_img_num = cat_img_num
        self.copys = copys
         
    def __len__(self):
        return int(len(self.mixup) / self.copys)

    def sampleList(self, sampleupper, sample_num):
        # choices -> with replacement; sample -> without replacement 
        return random.sample(range(sampleupper), k=sample_num)
    
    def eraseLambda1(self, index):
        img = self.mixup[index]
        info = self.auxi_info[index]
        return img/info['1_lambda']
           
    def __getitem__(self, index):
        cum_idx_lst = self.sampleList(self.cat_img_num, self.cat_img_num)
        real_idx = index * self.copys
        # image reweighting
        x_list = self.eraseLambda1(real_idx+cum_idx_lst[0]).unsqueeze(0)
      
        for i in cum_idx_lst[1:]:
            x_list = torch.cat( (  x_list, self.eraseLambda1(real_idx + i).unsqueeze(0)  ) )
        #################################################################
        reweight_x = torch.zeros_like(x_list)
        var_lst = []
        for i in range(self.cat_img_num):
            var_lst.append(x_list[i].var())
        var_ten = torch.tensor(var_lst)
        var_ten = var_ten / var_ten.min()
        # for each picture, divide variance
        for i in range(self.cat_img_num):
            reweight_x[i] = x_list[i] / var_ten[i]
        # sort, ascending order
        _, indices = torch.sort(var_ten)
        #x_list = x_list[indices]
        #################################################################
        return x_list, reweight_x, self.augmented_img[ real_idx + cum_idx_lst[indices[0]] ] 
    
class MultiImgDataset(Dataset):
    def __init__(self, cat_img_num, copys, limit, datasetname, parameterK, suffix): 
        catdata = ClusterDataset(cat_img_num, copys, limit, datasetname, parameterK, suffix)
        testloader = torch.utils.data.DataLoader(catdata, batch_size=len(catdata), shuffle=False)
        self.x, self.reweight_x, self.y = next(iter(testloader))
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.reweight_x[index], self.y[index]    

########################################## Utilities ##########################################

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

def computeLoss(yhat, y, alpha, L1_loss, ssim_loss):
    L1_out = L1_loss(yhat, y)
    ssim_out = -ssim_loss(yhat, y)
    final_loss = (1-alpha) * L1_out + alpha * ssim_out
    return final_loss, -ssim_out
    
########################################## Test Body ##########################################
def netForward(fcnn, rnan, x):
    if cat_num_image == 1:
        yhat = rnan(x.squeeze())
    elif cat_num_image > 1:
        yhat = rnan(fcnn( *[x[:, i] for i in range(cat_num_image)] ) )
    else:
        yhat = None 
    return yhat
   
def checktestaccur(fcnn, rnan, testdataloader, L1_loss, ssim_loss):
    rnan.eval()
    fcnn.eval()
    loss = float(0)
    ssim_ck = float(0)
    with torch.no_grad():
        for _, reweight_x, y in testdataloader:
            reweight_x = reweight_x.to(device)
            y = y.to(device)
            yhat = netForward(fcnn, rnan, reweight_x)
            batch_loss, ssim_bl =  computeLoss(yhat, y, rnan_alpha, L1_loss, ssim_loss)
            loss += batch_loss
            ssim_ck += ssim_bl
    return loss/len(testdataloader), ssim_ck/len(testdataloader)
    
if __name__=='__main__':   

    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    datasetname = "CELEBA"
    cat_num_image = 10
    auglimit = 16
    parameterK = 6
    suffix = ""
    
    print("Restoring Normal Pixels with Signs...")
    print("********* Dataset:", datasetname, "Cat image:", cat_num_image, "Auglimit:", auglimit, "*********")
    fuse = 0 if cat_num_image <= 10 else 2                   # MAX 0, AVG 2
    fscheme = "avg" if fuse == 2 else "max"
    copys = 50
    catdata = MultiImgDataset(cat_num_image, copys, auglimit, datasetname, parameterK, suffix)
    batch_sz = 64
    test_len = int(len(catdata) * 0.2)
    total_len = int(len(catdata))
    train_len = total_len - test_len
    trainset, testset = torch.utils.data.random_split(catdata, [train_len, test_len])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sz, shuffle=True)
    print("len(trainset):", len(trainset), "len(testset)", len(testset))
    print("len(trainloader):", len(trainloader), "len(testloader):", len(testloader))

    if cat_num_image > 1:
        epoches = 80
    else:
        epoches = 100
    rnan_alpha = 0.7

    # models
    fcnn = FusionNet(fuse_scheme=fuse).to(device)
    fcnn_optimizer = torch.optim.Adam(fcnn.parameters(), lr=1e-4) 

    rnan = RNAN(n_resgroup=6, n_feats=48, reduction=16, n_colors=3).to(device)
    rnan_optimizer = torch.optim.Adam(rnan.parameters(), lr=1e-4) 
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(rnan_optimizer)
    L1_loss = torch.nn.L1Loss(reduction='mean')
    ssim_loss = pytorch_ssim.SSIM(window_size = 8)

    train_losses = []
    test_losses = []

    start = time.time()

    for epoch in range(1, epoches+1):
        accurate = float(0)
        train_len = len(trainloader)
        sum_loss = 0 
       
        rnan.train()
     
        fcnn.train()
        for _, reweight_x, y in trainloader:
            reweight_x = reweight_x.to(device)
            y = y.to(device)
            
            yhat = netForward(fcnn, rnan, reweight_x)

            rnan_optimizer.zero_grad()
            fcnn_optimizer.zero_grad()
           
            batch_loss, _ = computeLoss(yhat, y, rnan_alpha, L1_loss, ssim_loss)
            batch_loss.backward() 
            rnan_optimizer.step()
            if cat_num_image > 1:
                fcnn_optimizer.step()
            sum_loss += batch_loss   
            # Learning rate scheduling should be applied after optimizer's update
            # scheduler.step(batch_loss)

        train_loss = sum_loss/len(trainloader)
        train_losses.append(train_loss)
        
        test_loss, ssim_test = checktestaccur(fcnn, rnan, testloader, L1_loss, ssim_loss )
        test_losses.append(test_loss)
        print("In epoch {}/{}, train total {}, total test loss is {}, SSIM test loss is {}.".format(
            epoch, epoches, train_loss, test_loss, ssim_test))
        print('%s (%d %d%%)' % (timeSince(start, epoch / epoches), epoch, epoch / epoches * 100))
       
    IDString = "AugLimit{}-Cat{}-k{}-normalTraining-epoch{}-{}-reweight-Normalpixel{}".format(auglimit, cat_num_image, parameterK, epoches, fscheme, suffix)
    torch.save(rnan.state_dict(), "{}-ImageFusion2RNAN-RNAN-{}".format(datasetname, IDString))
    if cat_num_image > 1:
        torch.save(fcnn.state_dict(), "{}-ImageFusion2RNAN-FCNN-{}".format(datasetname, IDString))

    torch.save(train_losses, "{}-train_losses-ImageFusion2RNAN-{}".format(datasetname, IDString))
    torch.save(test_losses, "{}-test_losses-ImageFusion2RNAN-{}".format(datasetname, IDString))

    
