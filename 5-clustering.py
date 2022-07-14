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

def computeSim(compnn, mixups, max_class, M):
    """ Compute the Similarity Matrix
    """
    start = time.time()
    simMatrix = torch.zeros(M, M)
    with torch.no_grad():
        for i in range(M):
            if i % 500 == 100:
                print("Processing {}th row...".format(i))
                print('%s (%d %d%%)' % (timeSince(start, i / M), i, i / M * 100))
            for j in range(M):
                if i == j: 
                    simMatrix[i][j] = 1 
                    continue
                elif i > j:  # lower trangle
                    simMatrix[i][j] = simMatrix[j][i]
                    continue
                elif max_class[i] != max_class[j]:
                    continue
                    
                x = torch.cat((mixups[i].unsqueeze(0), mixups[j].unsqueeze(0)), dim=0).unsqueeze(0)
                y = compnn(*[x[:, i] for i in [0, 1]]).squeeze()
                simMatrix[i][j] = y[1]
    return simMatrix
    
def insert(iset, no, M):
    if no <= 0:
        return iset
    dist = torch.zeros(M)
    for i in range(M):
        if i in iset:
            continue
        idist = 0
        for u in iset: 
            idist += simmatrix[i][u]
        dist[i] = idist
    candidate = dist.argmax()
    iset.add(candidate.item())
    return insert(iset, no - 1, M)

def create(M):
    """ for each mixup image, find a set of mixups that are most similar to it
    """
    set_lst = []
    start = time.time()
    for i in range(M):   
        testset = set()
        testset.add(i) 
        set_lst.append(insert(testset, 12, M))
        if i % 500 == 100:
            print("Processing {}th mixup image...".format(i)) 
            print('%s (%d %d%%)' % (timeSince(start, i / M), i, i / M * 100))
    return set_lst
    
def getdist(x, y):
    zi = len(x.intersection(y))
    zu = len(x.union(y)) 
    return zu/zi if zi != 0 else 1e6

def getdistMatrix(setlst):
    """ compute distance matrix based on the intersection of similar images of each pair of mixups 
    """
    distMatrix = torch.zeros(M, M)
    for i in range(M):
        if (i + 1) % 500 == 0:
            print("Processing", i, "rows...")
        for j in range(M):
            if i == j:
                continue
            distMatrix[i][j] = getdist(setlst[i], setlst[j])
    return distMatrix
    
def assignCenter(centers, distmatrix):
    set_dict = {}
    for c in centers:
        s = set()
        s.add(c)
        set_dict[c] = s
    for i in range(M): 
        if i in centers:
            continue
        temp_dist = 1e10
        assign_c = -1
        for c in centers:
            if distmatrix[i][c] < temp_dist:
                temp_dist = distmatrix[i][c]
                assign_c = c 
        cset = set_dict[assign_c]
        cset.add(i)
    return set_dict
    
def reassignCenter(centers, center_dict, distmatrix):
    newcenters = set()
    for c in centers:
        oldset = center_dict[c]
        temp_dist = 1e10 
        temp_center = -1
        for i in oldset:
            idist = 0
            for j in oldset:
                if i == j:
                    continue
                else:
                    idist += distmatrix[i][j]
            if idist < temp_dist:
                temp_dist = idist 
                temp_center = i 
        newcenters.add(temp_center)
    return newcenters
    
if __name__=='__main__':   

    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    
    mixups, labels = loadRawMixups("ToyData/toy_encryption.npy", "ToyData/toy_label.npy")    # toy samples
    
    M = len(mixups)    # num of mixups
    I = 2         # num of private images
    
    max_class = [] # the class with the max lambda 
    for i in range(len(labels)):
        label = labels[i]
        nonzero = (label!=0).sum()
        idx = label.argmax()
        max_class.append(idx)
        if nonzero == 2:
            maxv = label.max()
        elif nonzero == 1: 
            maxv = label.max()/2
        assert nonzero == 1 or nonzero == 2
        # re-weighting by lambda 
        mixups[i] = mixups[i]/maxv  
    
    compnn = ComparativeNN().to(device)
    compnn.load_state_dict(torch.load("ComparativeNetwork-50ep", map_location=device))
    
    simmatrix = computeSim(compnn, mixups, max_class, M)
    torch.save(simmatrix, "simmatrix")
    print("Simmatrix Finished!")  
    
    setlst = create(M)
    torch.save(setlst, "setlst")
    print("Setlst Finished!") 
    
    distMatrix = getdistMatrix(setlst)
    print("DistMatrix Finished!")  
    
    # clustering
    centers = random.sample(range(M), k=I)
    centers = set(centers)
    count = 1
    while True:
        center_dict = assignCenter(centers, distMatrix)
        newcenters = reassignCenter(centers, center_dict, distMatrix)
        inter = len(newcenters.intersection(centers))
        if inter == I: # no more change
            break 
        else:
            centers = newcenters
        print("Iterating {}, change {} centers".format(count, I - inter))
        count += 1
        
    torch.save(center_dict, "center_dict")
    torch.save(newcenters, "newcenters")
    print("Clustering Finished!")  
    