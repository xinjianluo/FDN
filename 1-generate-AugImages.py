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

# def getCIFAR100Tensors():
    # with open("test_batch_cifar_100", 'rb') as fo:
        # test = pickle.load(fo, encoding='bytes')
    # data_test = test[b'data']
    # labels_test = test[b'fine_labels'] 
    # data_test = np.array(data_test)
    # labels_test = np.array(labels_test)
    # print("Data test shape:", data_test.shape)
    # data_test = data_test.reshape((data_test.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    
    # data = data_test
    # labels = labels_test
    # print("For CIFAR100:", data.shape, labels.shape)
    # return data, labels, "CIFAR100"
    
# def getCIFAR10Tensors():
    # with open("test_batch_cifar_10", 'rb') as fo:
        # test = pickle.load(fo, encoding='bytes')
    # data_test = test[b'data']
    # labels_test = test[b'labels'] 
    # data_test = np.array(data_test)
    # labels_test = np.array(labels_test)
    # print("Data test shape:", data_test.shape)
    # data_test = data_test.reshape((data_test.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

    # data = data_test
    # labels = labels_test
    # print("For CIFAR10:", data.shape, labels.shape)
    # return data, labels, "CIFAR10"

# def getSTL10Tensors():
    # images = np.load("test_batch_stl10_x.npy")
    # labels = np.load("test_batch_stl10_y.npy")
    # print("For STL10, images shape {}, labels shape {}".format(images.shape, labels.shape))
    # return images, labels, "STL10"

def getCELEBATensors():
    images = np.load("ToyData/toy_celeba.npy")
    labels = np.asarray(range(len(images)))
    print("For CELEBA, images shape {}, labels shape {}".format(images.shape, labels.shape))
    return images, labels, "CELEBA"      
    
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
    
def totensor(img):
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [-1.0, 1.0]
    # (mean, std) is applicable to cifar normalization
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])(img)

def isNear(orig, augmented, limitvalue, image_sz):
    limitvalue = math.ceil(limitvalue/2)
    for h in range(limitvalue, image_sz - limitvalue):
        for w in range(limitvalue, image_sz - limitvalue):
            if random.uniform(0, 1) < 0.5:
                continue
            pixel = orig[:, h, w]
            flag = False
            for aug_h in range(h - limitvalue, h + limitvalue):
                if flag:
                    break
                for aug_w in range(w - limitvalue, w + limitvalue):
                    aug_pixel = augmented[:, aug_h, aug_w]
                    if (aug_pixel == pixel).sum() >=3:
                        flag = True
                        break            
            if flag == False:
                return False
    return True 
       
def getTensors(dataset_name):
    if dataset_name == "CELEBA":
        x_img, y_label, datasetname = getCELEBATensors()
    # elif dataset_name == "CIFAR100":
        # x_img, y_label, datasetname = getCIFAR100Tensors()
    # elif dataset_name == "CIFAR10":
        # x_img, y_label, datasetname = getCIFAR10Tensors()
    # elif dataset_name == "STL10":
        # x_img, y_label, datasetname = getSTL10Tensors()
    return x_img, y_label, datasetname

if __name__=='__main__':  

    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    image_sz = 32

    dataset_name = "CELEBA"
    epsilon = 0.5
    aug_num = 49
    
    x_img, y_label, datasetname = getTensors(dataset_name)
    position_limit = int(image_sz*epsilon)   
    mytransforms = [transforms.RandomAffine(90, translate=(0.50, 0.50)), transforms.RandomRotation(90), 
                    transforms.Compose([transforms.RandomCrop(image_sz-int(position_limit/2)), transforms.Resize(image_sz)])]

    print("dataset:", datasetname, "Epsilon:", epsilon, "position_limit:", position_limit)
    print("One image has additional", aug_num, "augmentations (except original image).")
    # for each image, execute data augmentation
    augmented_data = None
    augmented_labels = []
    augmented_class = []
    start = time.time()
    total = len(x_img)
    for i in range(len(x_img)):
        if i > 0:
            print('%s (%d %d%%)' % (timeSince(start, i / total), i, i / total * 100)) 
            print(i, "/", total, '->', augmented_data.shape)
            
        y = i 
        x = x_img[i]
        pilimg = transforms.ToPILImage()(x)
        orig_x = totensor(pilimg)
        # first add original pic 
        if augmented_data is None:
            augmented_data = orig_x.unsqueeze(dim=0).to(device)
        else:
            augmented_data = torch.cat((augmented_data, orig_x.unsqueeze(dim=0).to(device)))
        augmented_labels.append(y)
        augmented_class.append(y_label[i])
        
        # second add augmentated pics 
        # random.shuffle(trans_idx)
        count = 0 
        while(count < aug_num):
            for idx in range(len(mytransforms)):
                pil_aug_img = mytransforms[idx](pilimg)
                aug_img = totensor(pil_aug_img)
                if isNear(orig_x, aug_img, position_limit, image_sz) and aug_img.sum() != orig_x.sum():
                    augmented_data = torch.cat((augmented_data, aug_img.unsqueeze(dim=0).to(device)))
                    augmented_labels.append(y) 
                    augmented_class.append(y_label[i])
                    count += 1  
                    if count >= aug_num:
                        break 
            
    augmented_labels = torch.tensor(augmented_labels) 
    print("Shape of augmented_data:", augmented_data.shape, "; Shape of augmented_labels:", augmented_labels.shape)

    IDString = "limit{}-a{}".format(position_limit, aug_num + 1)
    torch.save(augmented_data.cpu(), '{}-AugmentatedImages-{}.pt'.format(datasetname, IDString))
    torch.save(augmented_labels.cpu(), '{}-AugmentatedImgIndexes-{}.pt'.format(datasetname, IDString))
    torch.save(augmented_class, '{}-AugmentatedImgClasses-{}.pt'.format(datasetname, IDString))






