import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import pickle
from sklearn.cluster import KMeans
import random
import torchvision.utils as tvutils
import numpy as np
import torchvision.models as tvmodels
import cv2

class CustomImageNet(Dataset):
    def __init__(self, filepath, imagesize=32):
        with open(filepath, 'rb') as fo:
            whole = pickle.load(fo)

        data = whole['data']
        labels = whole['labels']
        data = np.array(data)
        self.labels = torch.tensor(labels)
        self.data = data.reshape((data.shape[0], 3, imagesize, imagesize)).transpose(0, 2, 3, 1)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])           
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
            
        return sample, self.labels[index] 
    
    
def getRandomValues(k=6): 
    while True:
        a = np.random.random(k)
        a /= a.sum()
        a = sorted(a, reverse=True)
        if a[0] < 0.65 and a[0] + a[1] > 0.3:
            return a 
            
def getGaussianImg():
    # After totensor(), before normalize()
    # same mean and std as  [after imagenet totensor()]
    gx = torch.normal(mean=0.4355, std=0.2591, size=(3, 32, 32))
    gx = torch.clamp(gx, 0, 1)
    ngx = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))(gx)
    return ngx  

def mixgaussian(prix, priy, lambdas):
    result = prix * lambdas[0] + priy * lambdas[1]
    for lam in lambdas[2:]: 
        pub_img = getGaussianImg()
        pub_img = pub_img.to(device)
        result = result + pub_img * lam                
    return  result      
         
def mixpublic(prix, priy, lambdas, imagenet_kpset):
    result = prix * lambdas[0] + priy * lambdas[1]
    pub_max_idx = len(imagenet)
    for lam in lambdas[2:]:
        while True:
            pub_idx = random.randint(0, pub_max_idx-1)
            if(imagenet_kpset[pub_idx] >= 30):  # only mix those public images with > 30 key points
                pub_img, _ = imagenet[pub_idx]
                pub_img = pub_img.to(device)
                result = result + pub_img * lam
                break
    return  result    

def convert2positive(t):
    mask = (t > 0)*1 + (t < 0)*-1
    return mask * t 
    
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
        
if __name__=='__main__':          
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
           
    imagenet = CustomImageNet('ToyData/imagenet_val_data_2000')
    print("After creat imagenet dataset...")

    datasetname = "CELEBA"
    IDString = "limit16-a50"
    parameterK = 6

    print("*********************** Processing dataset", datasetname, "IDString:", IDString, "***********************")
    rootdir = ""
    augmented_data = torch.load('{}{}-AugmentatedImages-{}.pt'.format(rootdir, datasetname, IDString))
    augmented_labels = torch.load('{}{}-AugmentatedImgIndexes-{}.pt'.format(rootdir, datasetname, IDString))
    augmented_class = torch.load('{}{}-AugmentatedImgClasses-{}.pt'.format(rootdir, datasetname, IDString))

    pub_max_idx = len(imagenet)
    mixup = None
    auxi_info = []
    second_pic_idx = [i for i in range(len(augmented_data))]
    random.shuffle(second_pic_idx)

    print("Computing key point set of imagenet...")
    imagenet_kpset = []
    for i in range(len(imagenet)):
        pub_img, _ = imagenet[i]
        img = np.transpose(pub_img.numpy(), (1,2,0))
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        sift = cv2.xfeatures2d.SIFT_create()
        kp, _ = sift.detectAndCompute(gray, None)
        imagenet_kpset.append(len(kp))

    start = time.time()
    print("Generating mixup images...")
    for i in range(len(augmented_data)):
        if i % 50000 == 100:
            total = len(augmented_data)
            print('%s (%d %d%%)' % (timeSince(start, i / total), i, i / total * 100)) 
            print(i, "/", total, '->', mixup.shape)
            
        x_idx = i
        y_idx = second_pic_idx[i]
        x, x_label, x_class = augmented_data[x_idx], augmented_labels[x_idx], augmented_class[x_idx]
        y, y_label, y_class = augmented_data[y_idx], augmented_labels[y_idx], augmented_class[y_idx]
        lambdas = getRandomValues(k=parameterK)
        x = x.to(device)
        y = y.to(device)
        t_mix = mixpublic(x, y, lambdas, imagenet_kpset)
        #t_mix = mixgaussian(x, y, lambdas)
        t_mix = convert2positive(t_mix)     
        if mixup is None:
            mixup = t_mix.unsqueeze(dim=0)
        else:
            mixup = torch.cat([mixup, t_mix.unsqueeze(dim=0)])
        auxi_info.append({"idx":i, "1_idx":x_idx, "1_label":x_label.item(), "1_class":x_class, "1_lambda":lambdas[0], 
                                   "2_idx":y_idx, "2_label":y_label.item(), "2_class":y_class, "2_lambda":lambdas[1]})

    torch.save(mixup.cpu(), '{}-MixupPublicImages-k{}-{}.pt'.format(datasetname, parameterK, IDString))
    torch.save(auxi_info, '{}-public-auxi_info-k{}-{}.lst'.format(datasetname, parameterK, IDString))      

    print(mixup.shape, len(auxi_info))
    print(auxi_info[0])




