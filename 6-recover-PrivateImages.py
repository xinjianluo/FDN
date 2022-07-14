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
from torchvision.utils import save_image
from AttackModels import FusionNet, RNAN

def convert2positive(t):
    mask = (t > 0)*1 + (t < 0)*-1
    return mask * t
    
def loadRawMixups(mixup_path, mix_label_path):
    raw_mixups = np.load(mixup_path)
    print("In encryption.npy, max:", raw_mixups.max(), " ; min:", raw_mixups.min())
    mixups_np = raw_mixups.reshape((raw_mixups.shape[0], 32, 32, 3))
    mixups = torch.from_numpy(np.transpose(mixups_np, (0, 3, 1, 2)))
    mixups = convert2positive(mixups)
    
    raw_labels = np.load(mix_label_path)
    labels = torch.from_numpy(raw_labels)
    # torch.Size([5000, 3, 32, 32]) torch.Size([5000, 10])
    print("mixups.shape:", mixups.shape, "labels.shape:", labels.shape)
    return mixups, labels

    
if __name__=='__main__':   

    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    
    mixups, labels = loadRawMixups("ToyData/toy_encryption.npy", "ToyData/toy_label.npy")    # toy samples
    center_dict = torch.load("center_dict")
    newcenters = torch.load("newcenters")
    
    fcnn = FusionNet(fuse_scheme=0).to(device)
    rnan = RNAN(n_resgroup=6, n_feats=48, reduction=16, n_colors=3).to(device)
    fcnn.load_state_dict(torch.load("CELEBA-ImageFusion2RNAN-FCNN-AugLimit16-Cat10-k6-normalTraining-epoch80-max-reweight-Normalpixel", map_location=device))
    rnan.load_state_dict(torch.load("CELEBA-ImageFusion2RNAN-RNAN-AugLimit16-Cat10-k6-normalTraining-epoch80-max-reweight-Normalpixel", map_location=device))
    
    for target_id in newcenters:
        positive_lst = center_dict[target_id]
        x_list = mixups[target_id].unsqueeze(0)

        for i in positive_lst:
            if i == target_id:
                continue
            x_list = torch.cat( (  x_list, mixups[i].unsqueeze(0)  ) )

        cat_num_image = len(x_list)
        reweight_x = torch.zeros_like(x_list)
        var_lst = []
        for i in range(cat_num_image):
            var_lst.append(x_list[i].var())
        var_ten = torch.tensor(var_lst)
        var_ten = var_ten / var_ten.min()

        for i in range(cat_num_image):
            reweight_x[i] = x_list[i] / var_ten[i]
   
        reweight_x = reweight_x.unsqueeze(0)

        ymerge = fcnn(*[reweight_x[:, i] for i in range(cat_num_image)])

        yhat = rnan(ymerge)
        save_image(yhat.squeeze(), "private-{}.png".format(target_id))
        
    print("Attack Finished!")  
    